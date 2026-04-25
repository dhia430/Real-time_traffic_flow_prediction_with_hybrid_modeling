# Real-Time Traffic Flow Prediction Pipeline

> **Current Status**: 🚀 Training YOLOv8m on 8,202 images (100 Epochs) - In Progress.

A hybrid system that combines Computer Vision (YOLOv8) with Macroscopic Traffic Modeling (Cell Transmission Model - CTM) to monitor road traffic density and predict congestion shockwaves in real-time.

## Project Architecture
- **Computer Vision**: Detects and tracks vehicles in traffic camera feeds using YOLOv8.
- **Traffic Modeling**: Uses the Cell Transmission Model (CTM) to propagate observed densities across simulated road segments (cells).
- **Visualization**: Generates real-time dashboards overlays including bounding boxes, track IDs, ROI, and predicted congestion heatmaps.

## Folder Structure
```text
traffic_prediction/
├── config/             # YAML configuration files
├── data/               # Extracted images, labels, and splits
├── models/             # Trained YOLO weights (best.pt)
├── src/
│   ├── cv/             # YOLO detection and tracking
│   ├── traffic/        # CTM and density estimation
│   ├── utils/          # Data splitting and prep
│   ├── pipeline/       # Main execution scripts
│   └── visualization/  # Plotting and frame rendering
├── tests/              # Unit tests
├── outputs/            # Final result videos and plots
├── requirements.txt    # Python dependencies
└── run.bat             # Main launcher script (Windows)
```

## Setup Instructions

### 1. Prerequisites
- Python 3.9+ installed and added to PATH.

### 2. Installation
Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Unzip and split your `obj.zip` dataset into the YOLO-structured `train/val/test` folders:
```bash
python src/utils/data_splitter.py
```

### 4. Training (Optional)
If you have annotated images, you can fine-tune the YOLOv8 model:
```bash
python src/cv/train_yolo.py
```
*Trained weights will be saved to `models/saved_models/weights/best.pt`.*

## Running Inference
To run the full pipeline on a video file and save the result:
```bash
python src/pipeline/run_pipeline.py --video path/to/your/video.mp4 --output outputs/result.mp4
```

## Configuration Parameters (`config/config.yaml`)
- **ROI**: Define the 4-point polygon `roi_points` representing the road surface in your camera view.
- **Road Length**: `road_length_meters` defines the physical length covered by your ROI for density math.
- **CTM Constants**: Adjust `free_flow_speed_m_s`, `jam_density_veh_m`, and `max_flow_veh_s` to match the specific road type (highway vs. urban).

## Verification Scripts
Run these scripts to ensure components are working correctly:
- `verify_detection.py`: Runs detection on a subset of test images.
- `verify_density.py`: Processes a video and generates a density vs. time plot.
- `verify_ctm.py`: Simulates a shockwave burst and visualizes cell propagation.

## Troubleshooting
- **CFL Condition**: If a warning appears, CTM might be unstable. Ensure `(v_f * dt) / dx <= 1`.
- **YOLO Missing**: If `best.pt` is missing, the detector will automatically download a pretrained `yolov8n.pt`.
