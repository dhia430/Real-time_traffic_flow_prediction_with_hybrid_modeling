import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

class VehicleTracker:
    """
    Simple centroid-based vehicle tracker to maintain consist ID trajectory
    across frames using scipy's linear sum assignment algorithm.
    """
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {} # dict of ID -> { 'centroid': (x,y), 'bbox': [x1,y1,x2,y2], 'cls': cls }
        self.disappeared = {}
        self.centroids_history = {} # ID -> list of (x,y)
        self.history_size = 5 # Number of frames to keep for speed calculation
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid, bbox, cls):
        self.objects[self.next_object_id] = {'centroid': centroid, 'bbox': bbox, 'cls': cls, 'speed_kmh': 0.0}
        self.disappeared[self.next_object_id] = 0
        self.centroids_history[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.centroids_history:
            del self.centroids_history[object_id]

    def update(self, detections):
        """
        Update tracker with new detections based on centroid distances.
        
        Args:
            detections: List of [x1, y1, x2, y2, conf, cls]
            
        Returns:
            dict of tracked objects: { ID: {'centroid': c, 'bbox': b, 'cls': c} }
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_bboxes = []
        input_classes = []
        
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_bboxes.append([x1, y1, x2, y2])
            input_classes.append(cls)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i], input_classes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj['centroid'] for obj in self.objects.values()]

            # Compute distance matrix between existing object centroids and input centroids
            D = distance.cdist(np.array(object_centroids), input_centroids)

            # Hungarian algorithm for optimal assignment
            row_inds, col_inds = linear_sum_assignment(D)

            used_rows = set()
            used_cols = set()

            for row, col in zip(row_inds, col_inds):
                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id]['centroid'] = input_centroids[col]
                self.objects[obj_id]['bbox'] = input_bboxes[col]
                self.objects[obj_id]['cls'] = input_classes[col]
                self.disappeared[obj_id] = 0
                
                # Update history
                self.centroids_history[obj_id].append(input_centroids[col])
                if len(self.centroids_history[obj_id]) > self.history_size:
                    self.centroids_history[obj_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            # Check for disappeared objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

            # Check for new objects
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self._register(input_centroids[col], input_bboxes[col], input_classes[col])

        return self.objects

    def calculate_speeds(self, fps: float, meters_per_pixel: float):
        """
        Calculate speeds for all tracked objects based on their centroid history.
        """
        if fps <= 0 or meters_per_pixel <= 0:
            return
            
        for obj_id, obj in self.objects.items():
            history = self.centroids_history.get(obj_id, [])
            if len(history) < 2:
                continue
                
            # Distance between oldest and newest centroid in history
            c1 = history[0]
            c2 = history[-1]
            dist_px = np.linalg.norm(np.array(c1) - np.array(c2))
            
            frames_elapsed = len(history) - 1
            time_elapsed_s = frames_elapsed / fps
            
            if time_elapsed_s > 0:
                dist_m = dist_px * meters_per_pixel
                speed_m_s = dist_m / time_elapsed_s
                speed_kmh = speed_m_s * 3.6
                self.objects[obj_id]['speed_kmh'] = speed_kmh
