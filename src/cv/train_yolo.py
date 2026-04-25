import os
import yaml
import logging
import argparse
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(config_path: str, weights_path: str = None, data_yaml_override: str = None):
    """
    Trains or fine-tunes a YOLO model.
    - If weights_path is provided, fine-tunes from those weights.
    - If data_yaml_override is provided, trains on that specific dataset yaml
      instead of the one in config (useful for fine-tuning on new data).
    """
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    model_conf = config_data.get('model', {})
    dataset_conf = config_data.get('dataset', {})
    
    data_yaml = dataset_conf.get('data_yaml_path', 'data/data.yaml')
    
    # If a new dataset yaml is provided (fine-tuning on new data), use it
    if data_yaml_override and os.path.exists(data_yaml_override):
        data_yaml = data_yaml_override
        logging.info(f"Fine-tuning will use NEW dataset yaml: {data_yaml}")
    
    if not os.path.exists(data_yaml):
        logging.error(f"Data YAML {data_yaml} not found. Run data_splitter.py first.")
        return

    if weights_path and os.path.exists(weights_path):
        model_name = weights_path
        logging.info(f"Fine-tuning from existing weights: {model_name}")
    else:
        model_name = model_conf.get('name', 'yolov8n.pt')
        
    epochs = model_conf.get('epochs', 50) 
    batch_size = model_conf.get('batch_size', 16)
    imgsz = model_conf.get('image_size', 640)
    patience = model_conf.get('patience', 20)
    lr0 = model_conf.get('learning_rate', 0.001)
    weight_decay = model_conf.get('weight_decay', 0.0005)
    
    logging.info(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    
    logging.info(f"Starting training on {data_yaml} for {epochs} epochs...")
    
    # Train model with augmentation and optimized hyperparameters for better precision/recall
    device_id = '0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Targeting GPU hardware: {gpu_name} (PyTorch CUDA index 0)")
    else:
        logging.warning("No NVIDIA GPU detected! Falling back to CPU.")
        
    results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        lr0=lr0,
        lrf=0.01,                 # Final learning rate = lr0 * lrf (cosine schedule)
        weight_decay=weight_decay,
        optimizer='AdamW',        # AdamW generalises better than SGD for fine-tuning
        warmup_epochs=3,          # Gradual warmup to stabilise early training
        # --- Data Augmentation (key for precision/recall improvement) ---
        mosaic=1.0,               # Mosaic: combines 4 images — best mAP booster
        mixup=0.15,               # Mixup: blends 2 images — helps generalisation
        copy_paste=0.1,           # Copy-paste augmentation for small objects
        degrees=10.0,             # Random rotation ±10° (handles tilted cameras)
        translate=0.1,            # Random translation ±10%
        scale=0.5,                # Random scale ±50% (helps detect far/close cars)
        shear=2.0,                # Slight shear for perspective variety
        perspective=0.0005,       # Subtle perspective distortion
        flipud=0.0,               # No vertical flip (cars don't drive upside down)
        fliplr=0.5,               # Horizontal flip (left/right lane variety)
        hsv_h=0.015,              # Hue jitter (lighting variety)
        hsv_s=0.7,                # Saturation jitter (overcast vs sunny)
        hsv_v=0.4,                # Brightness jitter (night/day)
        erasing=0.4,              # Random erasing (handles occlusion)
        # --- Output ---
        project='models',
        name='saved_models',
        device=device_id,
        exist_ok=True,
        plots=True,
        val=True,                 # Always validate after each epoch
        save_period=10,           # Save checkpoint every 10 epochs
    )
    
    logging.info(f"Training completed. Best weights saved to models/saved_models/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    default='config/config.yaml', help='Path to config file')
    parser.add_argument('--weights',   default=None, help='Path to pre-trained weights for fine-tuning')
    parser.add_argument('--data_yaml', default=None, help='Override data.yaml path (for new dataset fine-tuning)')
    args = parser.parse_args()
    
    train_model(args.config, args.weights, args.data_yaml)
