import os
import cv2
import yaml
import logging
from src.cv.detector import VehicleDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_detection(config_path: str, image_dir: str, output_dir: str):
    """
    Runs the detector on images in image_dir and saves outputs to output_dir
    annotated with bounding boxes.
    """
    if not os.path.exists(config_path):
        logging.error(f"Config path invalid: {config_path}")
        return
        
    detector = VehicleDetector(config_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_dir):
        logging.error(f"Image directory missing: {image_dir}")
        return
        
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        logging.warning("No images found in directory.")
        return
        
    for img_name in images[:10]: # Limit to 10 for quick verification
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        detections = detector.detect(img)
        
        for x1, y1, x2, y2, conf, cls in detections:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"cls:{int(cls)} {conf:.2f}", (int(x1), int(y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
        save_path = os.path.join(output_dir, f"out_{img_name}")
        cv2.imwrite(save_path, img)
        logging.info(f"Saved detection result: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--images', default='data/extracted/test/images', help='Path to test images')
    parser.add_argument('--output', default='outputs/verification/detections', help='Output directory')
    args = parser.parse_args()
    
    verify_detection(args.config, args.images, args.output)
