import cv2
import numpy as np

class Visualizer:
    """
    Handles logging and overlaying bounding boxes, tracks, densities, and CTM state
    visibly onto the main video feed.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        vis_conf = config.get('visualization', {})
        self.bbox_color = tuple(vis_conf.get('bbox_color', [0, 255, 0]))
        self.text_color = tuple(vis_conf.get('text_color', [255, 255, 255]))
        self.roi_color = tuple(vis_conf.get('roi_color', [255, 0, 0]))
        self.show_ids = vis_conf.get('show_ids', True)
        self.show_roi = vis_conf.get('show_roi', False)  # Hidden by default
        self.congestion_colors = vis_conf.get('congestion_colors', {
            'free_flow': [0, 255, 0],
            'heavy_traffic': [0, 165, 255],
            'jam': [0, 0, 255]
        })
        
        cam_conf = config.get('camera', {})
        self.roi_relative_points = cam_conf.get('roi_points', [[0.2, 0.9], [0.8, 0.9], [0.6, 0.4], [0.4, 0.4]])
        self.roi_points = None
        
    def draw_detections(self, frame: np.ndarray, tracked_objects: dict) -> np.ndarray:
        """
        Draws active tracks and bounding boxes on frame.
        """
        for obj_id, data in tracked_objects.items():
            x1, y1, x2, y2 = map(int, data['bbox'])
            speed = data.get('speed_kmh', 0.0)
            
            # Speed color coding (BGR format for OpenCV)
            if speed > 60:
                box_color = (0, 0, 255) # Red
            elif speed > 30:
                box_color = (255, 0, 0) # Blue
            else:
                box_color = (0, 255, 0) # Green
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            if self.show_ids:
                label = f"ID: {obj_id} | {speed:.0f} km/h"
                cv2.putText(frame, label, (x1, max(y1-10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2)
        return frame

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Outlines the Region of Interest polygon mapping the "Camera View".
        Controlled by show_roi in config.yaml.
        """
        if not self.show_roi:
            # Still scale the points for density estimator, but don't draw
            if self.roi_points is None:
                h, w = frame.shape[:2]
                self.roi_points = np.array([
                    [int(x * w), int(y * h)]
                    for x, y in self.roi_relative_points
                ], dtype=np.int32)
            return frame

        if self.roi_points is None:
            h, w = frame.shape[:2]
            self.roi_points = np.array([
                [int(x * w), int(y * h)]
                for x, y in self.roi_relative_points
            ], dtype=np.int32)

        if len(self.roi_points) > 0:
            cv2.polylines(frame, [self.roi_points], isClosed=True, color=self.roi_color, thickness=2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.roi_points], self.roi_color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
        return frame
        
    def overlay_dashboard(self, frame: np.ndarray, current_density: float, ctm_states: np.ndarray) -> np.ndarray:
        """
        Adds HUD indicating the CTM predicted states per cell
        """
        hud_start_y = 30
        
        # Display Current Density
        cv2.putText(frame, f"Real-Time Density: {current_density:.4f} veh/m", 
                    (20, hud_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
        # Visualize CTM cells dynamically
        hud_start_y += 30
        cell_w = 40
        cell_h = 20
        start_x = 20
        
        cv2.putText(frame, "CTM Predictive Cells:", (start_x, hud_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        hud_start_y += 10
        
        for i, dens in enumerate(ctm_states):
            x = start_x + (i * (cell_w + 5))
            y = hud_start_y
            
            # Gradient coloring roughly based on congestion
            if dens < 0.05:
                color = tuple(self.congestion_colors['free_flow'])
            elif dens < 0.10:
                color = tuple(self.congestion_colors['heavy_traffic'])
            else:
                color = tuple(self.congestion_colors['jam'])
                
            cv2.rectangle(frame, (x, y), (x+cell_w, y+cell_h), color, -1)
            cv2.rectangle(frame, (x, y), (x+cell_w, y+cell_h), (0,0,0), 1)
            cv2.putText(frame, f"C{i+1}", (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
        return frame
