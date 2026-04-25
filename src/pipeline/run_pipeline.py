import os
import cv2
import logging
import argparse
import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.cv.detector import VehicleDetector
from src.cv.tracker import VehicleTracker
from src.traffic.density_estimator import DensityEstimator
from src.traffic.ctm_model import CellTransmissionModel
from src.visualization.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(video_path: str, config_path: str, output_path: str, max_frames: int = None):
    """
    Main execution pipeline uniting Vision, Density Estimation, Math Modeling, and Rendering.
    """
    if not os.path.exists(config_path):
        logging.error(f"Config path invalid: {config_path}")
        return
        
    logging.info("Initializing Pipeline Components...")
    try:
        detector = VehicleDetector(config_path)
        tracker = VehicleTracker(max_disappeared=15)
        density_estimator = DensityEstimator(config_path)
        ctm = CellTransmissionModel(config_path, num_cells=10)
        visualizer = Visualizer(config_path)
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        return

    # Video Setup
    if not video_path or not os.path.exists(video_path):
        # Allow '0' as a string for webcam if the user explicitly wants it
        if video_path != '0':
            logging.error(f"Video file not found or path is invalid: '{video_path}'")
            return
            
    # Open the video source (0 for webcam, path for file)
    source = 0 if video_path == '0' else video_path
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logging.error(f"Error opening video source: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Auto-increment output path to prevent overwriting
    base, ext = os.path.splitext(output_path)
    counter = 1
    while os.path.exists(output_path):
        output_path = f"{base}_{counter}{ext}"
        counter += 1
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30, (width, height))
    
    logging.info(f"Starting pipeline on {video_path}. Outputting to {output_path}")
    
    pbar = tqdm.tqdm(total=total_frames if total_frames > 0 else None)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if max_frames is not None and frame_count >= max_frames:
            break
            
        # Step 1: Detect Vehicles
        detections = detector.detect(frame)
        
        # Step 2: Track ID assignments
        tracked_objects = tracker.update(detections)
        
        # Calculate speeds
        if not hasattr(tracker, 'meters_per_pixel'):
            roi_rel = density_estimator.roi_relative_points
            min_y = min(p[1] for p in roi_rel) * height
            max_y = max(p[1] for p in roi_rel) * height
            roi_h = max(max_y - min_y, 1.0)
            tracker.meters_per_pixel = density_estimator.road_length / roi_h
            
        tracker.calculate_speeds(fps if fps > 0 else 30, tracker.meters_per_pixel)
        
        # Step 3: Density Calculation in ROI
        current_density = density_estimator.calculate_density(tracked_objects, width, height)
        
        # Step 4: Advance CTM Math
        ctm_states = ctm.update(current_density)
        
        # Step 5: Render Data
        vis_frame = visualizer.draw_roi(frame)
        vis_frame = visualizer.draw_detections(vis_frame, tracked_objects)
        vis_frame = visualizer.overlay_dashboard(vis_frame, current_density, ctm_states)
        
        out.write(vis_frame)
        
        # Show the video feed on screen
        cv2.imshow('Traffic Prediction Pipeline', vis_frame)
        
        # Check if user presses 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User requested to stop the video feed.")
            break
            
        pbar.update(1)
        frame_count += 1
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()
    
    logging.info(f"Pipeline finished processing. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='outputs/result.mp4', help='Output video')
    parser.add_argument('--max_frames', type=int, default=None, help='Limit processing to N frames')
    args = parser.parse_args()
    
    run_pipeline(args.video, args.config, args.output, args.max_frames)
