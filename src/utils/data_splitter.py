import os
import shutil
import zipfile
import random
import yaml
import logging
from pathlib import Path
import argparse
import cv2
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_directory_structure(base_dir: str):
    """
    Creates the necessary YOLO directory structure under base_dir.
    """
    base_path = Path(base_dir)
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            (base_path / split / sub).mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory structure at {base_path}")
    return base_path

def split_data(raw_zip_path: str, extract_dir: str, splits: tuple = (0.7, 0.2, 0.1)):
    """
    Extracts zip, finds image/txt pairs, and splits them into train/val/test directories.
    
    Args:
        raw_zip_path (str): Path to raw dataset zip.
        extract_dir (str): Destination for structured YOLO data.
        splits (tuple): (train, val, test) ratio sum to 1.0.
    """
    if not os.path.exists(raw_zip_path):
        logging.error(f"Dataset zip not found at {raw_zip_path}")
        return

    # Temporary extraction folder
    temp_dir = Path("temp_extract")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        logging.info(f"Extracting {raw_zip_path}...")
        with zipfile.ZipFile(raw_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # Find all .jpg files
        all_images = list(temp_dir.rglob("*.jpg"))
        logging.info(f"Found {len(all_images)} images.")
        
        # Configure filtering criteria
        blur_threshold = 100.0
        valid_class_ids = {0, 1, 2, 3, 4, 6} # 5 is pedestrian, 7 is other
        
        # Valid pairs Filter (Image + Label) + Blur checking + Class checking
        valid_pairs = []
        filtered_blur = 0
        filtered_classes = 0

        for img_path in all_images:
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                continue
                
            # 1. Check for blur
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < blur_threshold:
                filtered_blur += 1
                continue
                
            # 2. Check for valid vehicle classes
            has_vehicle = False
            valid_lines = []
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                class_id = int(parts[0])
                if class_id in valid_class_ids:
                    has_vehicle = True
                    valid_lines.append(line)
                    
            if not has_vehicle:
                filtered_classes += 1
                continue
                
            # Rewrite the text file to only contain the valid lines
            with open(txt_path, 'w') as f:
                f.writelines(valid_lines)

            valid_pairs.append((img_path, txt_path))
            
        logging.info(f"Filtered {filtered_blur} blurry images.")
        logging.info(f"Filtered {filtered_classes} images with no vehicles.")
                
        if not valid_pairs:
            logging.error("No valid image/label pairs found after filtering!")
            return
            
        logging.info(f"Found {len(valid_pairs)} complete pairs. Shuffling...")
        random.shuffle(valid_pairs)
        
        # Calculate splits
        n_total = len(valid_pairs)
        n_train = int(n_total * splits[0])
        n_val = int(n_total * splits[1])
        
        train_pairs = valid_pairs[:n_train]
        val_pairs = valid_pairs[n_train:n_train+n_val]
        test_pairs = valid_pairs[n_train+n_val:]
        
        dest_base = create_directory_structure(extract_dir)
        
        def copy_files(pairs, split_name):
            logging.info(f"Copying {len(pairs)} pairs to {split_name}...")
            for img, txt in pairs:
                shutil.copy(img, dest_base / split_name / "images" / img.name)
                shutil.copy(txt, dest_base / split_name / "labels" / txt.name)

        copy_files(train_pairs, 'train')
        copy_files(val_pairs, 'val')
        copy_files(test_pairs, 'test')
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
def merge_and_split(zip1: str, zip2: str, extract_dir: str, splits: tuple):
    """
    Extracts two zip files into one temp folder, filters, merges, then splits.
    """
    temp_dir = Path("temp_extract_merged")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        for i, zip_path in enumerate([zip1, zip2], 1):
            if not os.path.exists(zip_path):
                logging.error(f"Zip {i} not found: {zip_path}")
                continue
            logging.info(f"Extracting zip {i}: {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir / f"zip{i}")
        
        # Find all images from both zips
        all_images = list(temp_dir.rglob("*.jpg")) + list(temp_dir.rglob("*.png"))
        logging.info(f"Total images found across both datasets: {len(all_images)}")
        
        blur_threshold = 100.0
        valid_class_ids = {0, 1, 2, 3, 6}  # car, motorcycle, bus, truck, van — NO pedestrians
        
        valid_pairs = []
        filtered_blur = 0
        filtered_classes = 0

        for img_path in all_images:
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < blur_threshold:
                filtered_blur += 1
                continue
            has_vehicle = False
            valid_lines = []
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                class_id = int(parts[0])
                if class_id in valid_class_ids:
                    has_vehicle = True
                    valid_lines.append(line)
            if not has_vehicle:
                filtered_classes += 1
                continue
            with open(txt_path, 'w') as f:
                f.writelines(valid_lines)
            valid_pairs.append((img_path, txt_path))
        
        logging.info(f"Filtered {filtered_blur} blurry + {filtered_classes} no-vehicle images.")
        logging.info(f"Merged valid pairs: {len(valid_pairs)}. Shuffling...")
        random.shuffle(valid_pairs)
        
        n_total = len(valid_pairs)
        n_train = int(n_total * splits[0])
        n_val   = int(n_total * splits[1])
        train_pairs = valid_pairs[:n_train]
        val_pairs   = valid_pairs[n_train:n_train+n_val]
        test_pairs  = valid_pairs[n_train+n_val:]
        
        dest_base = create_directory_structure(extract_dir)
        
        def copy_files(pairs, split_name):
            logging.info(f"Copying {len(pairs)} pairs to {split_name}...")
            for img, txt in pairs:
                shutil.copy(img, dest_base / split_name / "images" / img.name)
                shutil.copy(txt, dest_base / split_name / "labels" / txt.name)
        
        copy_files(train_pairs, 'train')
        copy_files(val_pairs,   'val')
        copy_files(test_pairs,  'test')
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def generate_data_yaml(dest_yaml: str, data_dir: str):
    """
    Generates data.yaml for YOLOv8.
    Only includes the 5 vehicle classes — pedestrians and 'other' are excluded.
    """
    data = {
        'train': os.path.abspath(os.path.join(data_dir, 'train', 'images')),
        'val':   os.path.abspath(os.path.join(data_dir, 'val',   'images')),
        'test':  os.path.abspath(os.path.join(data_dir, 'test',  'images')),
        'nc': 8,
        'names': ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'pedestrian', 'van', 'other']
    }
    
    os.makedirs(os.path.dirname(dest_yaml), exist_ok=True)
    with open(dest_yaml, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    logging.info(f"Generated YAML configuration at {dest_yaml}")

def main():
    parser = argparse.ArgumentParser(description="Split dataset.")
    parser.add_argument('--config',   default='config/config.yaml', help='Path to config file')
    parser.add_argument('--raw_zip',  default=None, help='Path to first (or only) raw zip')
    parser.add_argument('--raw_zip2', default=None, help='Path to second zip to merge with the first')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logging.error(f"Config file missing: {args.config}")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    ds_conf = config.get('dataset', {})
    train_s = ds_conf.get('train_split', 0.7)
    val_s   = ds_conf.get('val_split',   0.2)
    test_s  = ds_conf.get('test_split',  0.1)

    if args.raw_zip and args.raw_zip2:
        # --- MERGE two datasets into one timestamped folder ---
        timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
        extract_dir = f"data/extracted_{timestamp}"
        yaml_path   = f"data/data_{timestamp}.yaml"
        logging.info(f"Merging TWO datasets into: {extract_dir}")
        merge_and_split(args.raw_zip, args.raw_zip2, extract_dir, splits=(train_s, val_s, test_s))
        generate_data_yaml(yaml_path, extract_dir)
        with open('data/.last_new_data_yaml.txt', 'w') as f:
            f.write(os.path.abspath(yaml_path))
        logging.info(f"Merged dataset ready. Use fine-tuning option 2 to train on it.")

    elif args.raw_zip:
        # --- Single new dataset: extract to a fresh timestamped folder ---
        timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
        extract_dir = f"data/extracted_{timestamp}"
        yaml_path   = f"data/data_{timestamp}.yaml"
        logging.info(f"New dataset detected. Extracting to: {extract_dir}")
        split_data(args.raw_zip, extract_dir, splits=(train_s, val_s, test_s))
        generate_data_yaml(yaml_path, extract_dir)
        with open('data/.last_new_data_yaml.txt', 'w') as f:
            f.write(os.path.abspath(yaml_path))
        logging.info(f"New dataset ready. Use fine-tuning option 2 to train on it.")

    else:
        # --- Existing dataset: re-use original extracted folder ---
        extract_dir = ds_conf.get('extract_dir', 'data/extracted')
        yaml_path   = ds_conf.get('data_yaml_path', 'data/data.yaml')
        raw_zip     = ds_conf.get('raw_zip_path', '../wtp4ssmwsd-1/obj.zip')
        logging.info(f"Using existing dataset at: {extract_dir}")
        split_data(raw_zip, extract_dir, splits=(train_s, val_s, test_s))
        generate_data_yaml(yaml_path, extract_dir)

if __name__ == "__main__":
    main()
