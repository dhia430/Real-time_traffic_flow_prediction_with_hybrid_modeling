import os
import cv2
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_long_path(path):
    abs_path = os.path.abspath(path)
    if os.name == 'nt' and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path

def add_video_data(video_path: str, model_path: str, output_dir: str, interval: float = 2.0, conf: float = 0.5):
    """
    Extracts frames from video and auto-labels them using the provided model.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video not found: {video_path}")
        return
        
    if not os.path.exists(model_path):
        logging.error(f"Model weights not found: {model_path}. Train a model first!")
        return

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_interval = int(fps * interval)
    
    video_stem = Path(video_path).stem
    img_out = Path(get_long_path(os.path.join(output_dir, "images")))
    lbl_out = Path(get_long_path(os.path.join(output_dir, "labels")))
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    count = 0
    saved_count = 0
    
    logging.info(f"Extracting and auto-labeling frames from {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if count % frame_interval == 0:
            # Predict
            results = model.predict(frame, conf=conf, verbose=False)
            
            # Save if objects found
            if len(results[0].boxes) > 0:
                img_name = f"v_{video_stem}_{saved_count:05d}.jpg"
                img_file = img_out / img_name
                lbl_file = lbl_out / f"{Path(img_name).stem}.txt"
                
                # Save Image
                cv2.imwrite(str(img_file), frame)
                
                # Save Labels in YOLO format
                with open(str(lbl_file), 'w') as f:
                    for box in results[0].boxes:
                        # box.cls, box.xywhn
                        cls = int(box.cls[0])
                        xywhn = box.xywhn[0].tolist()
                        line = f"{cls} {' '.join(map(str, xywhn))}\n"
                        f.write(line)
                
                saved_count += 1
                if saved_count % 10 == 0:
                    logging.info(f"Saved {saved_count} auto-labeled frames...")
            
        count += 1
        
    cap.release()
    logging.info(f"Done! {saved_count} new samples added to {output_dir}")
    logging.info("Run Option 1 (re-split) in run.bat to include these in the next training run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--model', default='models/saved_models/weights/best.pt', help='Path to model weights for auto-labeling')
    parser.add_argument('--output', default='data/extracted/video_data', help='Output directory for new samples')
    parser.add_argument('--interval', type=float, default=2.0, help='Seconds between frame extractions')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for auto-labeling')
    args = parser.parse_args()
    
    add_video_data(args.video, args.model, args.output, args.interval, args.conf)
