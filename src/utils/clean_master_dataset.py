import os
import shutil
import zipfile
import random
import yaml
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_long_path(path):
    """Adds the Windows long path prefix if necessary."""
    abs_path = os.path.abspath(path)
    if os.name == 'nt' and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path

def create_directory_structure(base_dir: str):
    base_path = Path(get_long_path(base_dir))
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            (base_path / split / sub).mkdir(parents=True, exist_ok=True)
    return base_path

def filter_and_collect(temp_dir_str: str, source_id: int):
    """
    Filters images by blur and class.
    """
    temp_dir = Path(temp_dir_str)
    all_images = list(temp_dir.rglob("*.jpg")) + list(temp_dir.rglob("*.png"))
    logging.info(f"Source {source_id}: Found {len(all_images)} raw images.")
    
    blur_threshold = 100.0
    valid_class_ids = {0, 1, 2, 3, 4, 6}
    roboflow_map = {0: 4, 1: 2, 2: 0, 3: 1, 4: 3}
    
    valid_pairs = []
    filtered_blur = 0
    filtered_classes = 0

    for img_path in all_images:
        # Ensure we use long paths for file operations
        img_path_str = get_long_path(str(img_path))
        txt_path = Path(img_path_str).with_suffix('.txt')
        
        if not txt_path.exists():
            # Try finding labels in a sibling directory
            if img_path.parent.name == 'images':
                potential_txt = Path(get_long_path(str(img_path.parent.parent / 'labels' / img_path.with_suffix('.txt').name)))
                if potential_txt.exists():
                    txt_path = potential_txt
        
        if not txt_path.exists():
            continue
            
        # 1. Blur Filter
        img = cv2.imdecode(np.fromfile(img_path_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < blur_threshold:
            filtered_blur += 1
            continue
            
        # 2. Class Filter & Remapping
        has_target = False
        valid_lines = []
        try:
            with open(str(txt_path), 'r') as f:
                lines = f.readlines()
        except Exception: continue
        
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            try:
                class_id = int(parts[0])
            except ValueError: continue
            
            if source_id == 2 and class_id in roboflow_map:
                class_id = roboflow_map[class_id]
            
            if class_id in valid_class_ids:
                has_target = True
                parts[0] = str(class_id)
                valid_lines.append(" ".join(parts) + "\n")
        
        if not has_target:
            filtered_classes += 1
            continue
            
        # Save cleaned labels
        with open(str(txt_path), 'w') as f:
            f.writelines(valid_lines)
            
        valid_pairs.append((img_path_str, str(txt_path)))
        
    logging.info(f"Source {source_id}: Kept {len(valid_pairs)} pairs. Filtered {filtered_blur} blur, {filtered_classes} no-target.")
    return valid_pairs

def extract_frames_from_video(video_path: str, output_dir: str, interval_seconds: float = 1.0):
    """
    Extracts frames from a video at a given interval.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video not found: {video_path}")
        return []
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_interval = int(fps * interval_seconds)
    
    video_stem = Path(video_path).stem
    out_path = Path(get_long_path(output_dir))
    out_path.mkdir(parents=True, exist_ok=True)
    
    extracted_images = []
    count = 0
    saved_count = 0
    
    logging.info(f"Extracting frames from {video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if count % frame_interval == 0:
            img_name = f"v_{video_stem}_{saved_count:05d}.jpg"
            img_file = out_path / img_name
            cv2.imwrite(str(img_file), frame)
            extracted_images.append(str(img_file))
            saved_count += 1
            
        count += 1
        
    cap.release()
    logging.info(f"Extracted {saved_count} frames from video.")
    return extracted_images

def main():
    zip1 = "../wtp4ssmwsd-1.zip"
    zip2 = "../Yolov8 Traffic.v1i.yolov8.zip"
    master_dir = "data/master_dataset"
    master_yaml_path = "data/master_dataset.yaml"
    
    # Optional: Extract from videos if they exist
    video_sources = ["outputs/real_traffic_sample.mp4"] 
    
    if os.path.exists(master_dir):
        shutil.rmtree(master_dir)
    dest_base = create_directory_structure(master_dir)
    
    temp_extract_base = get_long_path("t_master_extract")
    if os.path.exists(Path(temp_extract_base)): 
        shutil.rmtree(Path(temp_extract_base))
    os.makedirs(temp_extract_base)
    
    all_valid_pairs = []
    
    try:
        # 1. Process ZIPs
        for i, zpath in enumerate([zip1, zip2], 1):
            if not os.path.exists(zpath): continue
            logging.info(f"Processing Source {i}...")
            s_temp = Path(temp_extract_base) / f"s{i}"
            with zipfile.ZipFile(os.path.abspath(zpath), 'r') as zf:
                zf.extractall(s_temp)
            # Handle nested
            if i == 1:
                nested = s_temp / "obj.zip"
                if nested.exists():
                    with zipfile.ZipFile(os.path.abspath(str(nested)), 'r') as nzf:
                        nzf.extractall(s_temp)
                    os.remove(str(nested))
            
            all_valid_pairs.extend(filter_and_collect(str(s_temp), i))
            
        # 2. Process Videos (If user wants to expand)
        # Note: Videos won't have labels, so they are mostly for manual check or future pseudo-labeling.
        # For now, we only merge labeled images from zips as requested for training.
        
        if not all_valid_pairs:
            logging.error("No valid pairs found!")
            return
            
        random.shuffle(all_valid_pairs)
        n = len(all_valid_pairs)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_p = all_valid_pairs[:n_train]
        val_p = all_valid_pairs[n_train:n_train+n_val]
        test_p = all_valid_pairs[n_train+n_val:]
        
        def move_pairs(pairs, split):
            logging.info(f"Copying {len(pairs)} pairs to {split}...")
            for img_path, txt_path in pairs:
                p_img = Path(img_path)
                new_name = f"{p_img.stem}_{split}_{random.getrandbits(16)}{p_img.suffix}"
                shutil.copy(img_path, dest_base / split / "images" / new_name)
                shutil.copy(txt_path, dest_base / split / "labels" / Path(new_name).with_suffix('.txt').name)

        move_pairs(train_p, 'train')
        move_pairs(val_p, 'val')
        move_pairs(test_p, 'test')
        
        data_yaml = {
            'train': str(dest_base / 'train' / 'images'),
            'val': str(dest_base / 'val' / 'images'),
            'test': str(dest_base / 'test' / 'images'),
            'nc': 8,
            'names': ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'pedestrian', 'van', 'other']
        }
        with open(master_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
            
        logging.info(f"Master dataset created with {n} total images.")

    finally:
        # Cleanup with long path support
        if os.path.exists(Path(temp_extract_base)):
            try:
                shutil.rmtree(Path(temp_extract_base))
            except Exception:
                logging.warning("Cleanup failed due to long paths, but data is ready.")

if __name__ == "__main__":
    main()
