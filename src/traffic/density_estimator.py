import cv2
import yaml
import numpy as np

class DensityEstimator:
    """
    Estimates traffic density by evaluating tracked objects within a 
    specific polygonal Region of Interest (ROI) and counting vehicles per frame.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        cam_conf = config.get('camera', {})
        self.roi_relative_points = cam_conf.get('roi_points', [[0.2, 0.9], [0.8, 0.9], [0.6, 0.4], [0.4, 0.4]])
        self.road_length = cam_conf.get('road_length_meters', 100.0)
        
        # Simple moving average state
        self.density_history = []
        self.sma_window_size = 5
        self.roi_points = None
        
    def _scale_roi(self, width: int, height: int):
        if self.roi_points is None:
            self.roi_points = np.array([
                [int(x * width), int(y * height)]
                for x, y in self.roi_relative_points
            ], dtype=np.int32)

    def _is_in_roi(self, centroid: tuple) -> bool:
        """Checks whether a point is within the defined ROI polygon"""
        if self.roi_points is None:
            return False
        point = (float(centroid[0]), float(centroid[1]))
        # pointPolygonTest returns >0 if inside, 0 if on edge, <0 if outside
        return cv2.pointPolygonTest(self.roi_points, point, False) >= 0

    def calculate_density(self, tracked_objects: dict, frame_width: int, frame_height: int) -> float:
        """
        Given the tracker output, determine how many vehicles are inside the ROI,
        and calculate density (vehicles per meter).
        
        Args:
            tracked_objects: dict of {id: {'centroid': (x,y), ...}}
            
        Returns:
            float: Smoothed density (vehicles / meter)
        """
        self._scale_roi(frame_width, frame_height)
        
        count_in_roi = 0
        for obj_id, data in tracked_objects.items():
            if self._is_in_roi(data['centroid']):
                count_in_roi += 1
                
        raw_density = count_in_roi / float(self.road_length)
        
        # Apply smoothing
        self.density_history.append(raw_density)
        if len(self.density_history) > self.sma_window_size:
            self.density_history.pop(0)
            
        smoothed_density = sum(self.density_history) / len(self.density_history)
        
        return smoothed_density
