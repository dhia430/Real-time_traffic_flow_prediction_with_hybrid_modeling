import os
import yaml
import logging
import numpy as np
from ultralytics import YOLO
import cv2
import torch
class VehicleDetector:
    """
    Wrapper for YOLOv8 model for vehicle detection.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config config.yaml not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        model_conf = config.get('model', {})
        self.model_path = model_conf.get('name', 'yolov8n.pt')
        self.conf_threshold = model_conf.get('confidence_threshold', 0.3)
        self.iou_threshold = model_conf.get('iou_threshold', 0.45)
        self.target_classes = model_conf.get('classes', [2, 3, 5, 7]) # COCO: car, motorcycle, bus, truck
        
        # Check if saved model exists from training, use it, else default back to model_path
        trained_model = os.path.join('models', 'saved_models', 'weights', 'best.pt')
        if os.path.exists(trained_model):
            self.model_path = trained_model
            logging.info(f"Loading trained weights from {trained_model}")
            # If using custom best.pt, force the use of our dataset classes
            # 0: car, 1: motorcycle, 2: bus, 3: truck, 6: van
            self.target_classes = [0, 1, 2, 3, 6]
        else:
            logging.info(f"Loading pretrained weights: {self.model_path}")
            
        self.model = YOLO(self.model_path)
        
        # Minimum bounding box area in pixels to be considered a real vehicle.
        # Lowered from 2000 to 500 to catch smaller/distant vehicles.
        self.min_bbox_area = 500  
        
    def detect(self, frame: np.ndarray):
        """
        Runs inference on a single frame.
        
        Args:
            frame (np.ndarray): OpenCV image frame.
            
        Returns:
            list: Parsed bounding boxes in format [x1, y1, x2, y2, conf, cls]
        """
        # Run YOLO inference
        device_id = '0' if torch.cuda.is_available() else 'cpu'
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=device_id,
            verbose=False
        )
        
        detections = []
        raw_count = len(results[0].boxes) if len(results) > 0 else 0
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                
                w = x2 - x1
                h = y2 - y1
                area = w * h
                
                # --- Filter 1: Minimum size (removes small noise/distant blobs) ---
                if area < self.min_bbox_area:
                    continue
                
                # --- Filter 2: Aspect ratio (removes vertical objects) ---
                if w > 0:
                    aspect_ratio = h / w
                    # Bus (2) and Truck (3) can be taller, but Cars (0) and Vans (6) should be wider.
                    # This helps distinguish vehicles from poles/trees.
                    # Relaxed limits for high-angle perspective distortion
                    if cls in [0, 6] and aspect_ratio > 1.5:
                        continue
                    if cls in [2, 3] and aspect_ratio > 2.0:
                        continue
                    if aspect_ratio > 3.0: # Extreme verticality is never a vehicle
                        continue
                        
                detections.append([x1, y1, x2, y2, conf, cls])
                
        return detections
