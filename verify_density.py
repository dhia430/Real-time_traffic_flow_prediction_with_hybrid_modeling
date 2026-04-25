import os
import cv2
import yaml
import logging
import matplotlib.pyplot as plt
from src.cv.detector import VehicleDetector
from src.cv.tracker import VehicleTracker
from src.traffic.density_estimator import DensityEstimator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_density(video_path: str, config_path: str, output_plot: str):
    """
    Processes a video and generates a density vs. time plot.
    """
    if not os.path.exists(config_path):
        logging.error(f"Config path invalid: {config_path}")
        return
        
    detector = VehicleDetector(config_path)
    tracker = VehicleTracker()
    estimator = DensityEstimator(config_path)
    
    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        logging.error(f"Error opening video: {video_path}")
        return
        
    densities = []
    
    logging.info("Processing video for density plot...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        den = estimator.calculate_density(tracked)
        densities.append(den)
        
        if len(densities) % 50 == 0:
            logging.info(f"Processed {len(densities)} frames...")
            
    cap.release()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(densities)
    plt.title("Traffic Density Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Density (veh/m)")
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot)
    plt.close()
    logging.info(f"Density plot saved to {output_plot}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--output', default='outputs/verification/density_plot.png', help='Output plot path')
    args = parser.parse_args()
    
    verify_density(args.video, args.config, args.output)
