import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

class VehicleTracker:
    """
    Advanced vehicle tracker with Per-Instance MPP Estimation.

    Instead of relying on a global linear perspective model (which only fits
    foreground vehicles and fails for far-away vehicles), this tracker computes
    meters-per-pixel INDIVIDUALLY for each vehicle at each frame using its
    own bounding box size as the reference.

    Reference dimensions used:
      - Close/street cameras  (bbox_h > 22% of frame): car HEIGHT ~ 1.5m
      - Aerial/elevated cameras (bbox_h <= 22% of frame): car WIDTH  ~ 1.8m
    """

    # Real-world vehicle reference dimensions (metres)
    _CAR_HEIGHT_M = 1.5   # Close/street camera  → use bbox HEIGHT
    _CAR_LENGTH_M = 4.0   # Aerial/elevated camera → use bbox WIDTH
    #   Elevated cameras show the car's SIDE profile, so bbox width ≈ car length

    def __init__(self, max_disappeared=30, max_distance=50, smoothing_factor=0.35):
        self.next_object_id = 0
        self.objects        = {}  # id -> {centroid, bbox, cls, speed_kmh}
        self.disappeared    = {}
        # History stores tuples of (bx, by, bbox_w_px, bbox_h_px)
        self.centroids_history = {}
        self.history_size   = 15
        self.max_disappeared = max_disappeared
        self.max_distance   = max_distance

        # Speed params
        self.smoothing_factor   = smoothing_factor
        self.min_history_points = 8     # Require 8 frames (~0.27s at 30fps)
        self.min_movement_px    = 1.5   # Noise floor — stationary guard handles stillness
        self.realistic_speed_max = 80   # Urban traffic cap km/h

        # Camera type detected from first frames
        self.camera_type  = None   # 'close' or 'aerial'
        self._frame_h     = None
        self._frame_w     = None
        self._bbox_samples = []    # list of bbox_h_px to detect camera type

        # Speed history per vehicle
        self.speed_history = {}

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def _register(self, centroid, bbox, cls):
        self.objects[self.next_object_id] = {
            'centroid': centroid, 'bbox': bbox, 'cls': cls, 'speed_kmh': 0.0
        }
        self.disappeared[self.next_object_id] = 0
        bx = int((bbox[0] + bbox[2]) / 2.0)
        by = int(bbox[3])
        bw = abs(bbox[2] - bbox[0])
        bh = abs(bbox[3] - bbox[1])
        self.centroids_history[self.next_object_id] = [(bx, by, bw, bh)]
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        self.centroids_history.pop(object_id, None)
        self.speed_history.pop(object_id, None)

    # ------------------------------------------------------------------ #
    # Camera-type detection                                                #
    # ------------------------------------------------------------------ #

    def _update_camera_type(self):
        """
        Decide close vs aerial from the first 60+ bbox height samples.
        Threshold: if median bbox_h > 22% of frame height → close camera.
        """
        if self.camera_type is not None or len(self._bbox_samples) < 60:
            return
        frame_h = self._frame_h or 720
        median_h_ratio = float(np.median(self._bbox_samples)) / frame_h
        self.camera_type = 'close' if median_h_ratio > 0.22 else 'aerial'
        print(f"[Camera-Type] Detected '{self.camera_type}' "
              f"(median bbox_h/frame_h = {median_h_ratio:.3f})")

    def _get_mpp_from_bbox(self, bbox_w_px, bbox_h_px):
        """
        Per-instance meters-per-pixel using the vehicle's own bbox size.
        No safety factor — bbox dimension directly encodes real-world scale.
        Close cameras  → HEIGHT ref (1.5m): stable from street-level side view
        Aerial cameras → LENGTH ref (4.0m): elevated cameras see car side profile,
                          bbox WIDTH ≈ car length (~4m), not car width (1.8m)
        """
        if self.camera_type == 'close' and bbox_h_px > 5:
            return self._CAR_HEIGHT_M / bbox_h_px
        elif bbox_w_px > 5:
            return self._CAR_LENGTH_M / bbox_w_px
        return 0.005  # safe fallback

    # ------------------------------------------------------------------ #
    # Main update loop                                                     #
    # ------------------------------------------------------------------ #

    def update(self, detections, frame_h=None, frame_w=None):
        if frame_h:
            self._frame_h = frame_h
        if frame_w:
            self._frame_w = frame_w

        # Collect bbox samples for camera-type detection
        for d in detections:
            if len(d) >= 4:
                h = abs(d[3] - d[1])
                if h > 5:
                    self._bbox_samples.append(h)
        self._update_camera_type()

        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_bboxes    = []
        input_classes   = []

        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_bboxes.append([x1, y1, x2, y2])
            input_classes.append(cls)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i], input_classes[i])
        else:
            object_ids       = list(self.objects.keys())
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            D                = distance.cdist(np.array(object_centroids), input_centroids)
            row_inds, col_inds = linear_sum_assignment(D)

            used_rows, used_cols = set(), set()
            for row, col in zip(row_inds, col_inds):
                if D[row, col] > self.max_distance:
                    continue
                obj_id = object_ids[row]
                self.objects[obj_id].update({
                    'centroid': input_centroids[col],
                    'bbox':     input_bboxes[col],
                    'cls':      input_classes[col],
                })
                self.disappeared[obj_id] = 0

                bx = int((input_bboxes[col][0] + input_bboxes[col][2]) / 2.0)
                by = int(input_bboxes[col][3])
                bw = abs(input_bboxes[col][2] - input_bboxes[col][0])
                bh = abs(input_bboxes[col][3] - input_bboxes[col][1])
                self.centroids_history[obj_id].append((bx, by, bw, bh))
                if len(self.centroids_history[obj_id]) > self.history_size:
                    self.centroids_history[obj_id].pop(0)
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])) - used_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            for col in set(range(D.shape[1])) - used_cols:
                self._register(input_centroids[col], input_bboxes[col], input_classes[col])

        return self.objects

    # ------------------------------------------------------------------ #
    # Speed calculation                                                    #
    # ------------------------------------------------------------------ #

    def calculate_speeds(self, fps: float, manual_mpp: float = None):
        if fps <= 0:
            return

        for obj_id in self.objects:
            history = self.centroids_history.get(obj_id, [])
            if len(history) < self.min_history_points:
                continue

            speed_kmh = self._instantaneous_speed(history, fps)
            if speed_kmh is None:
                continue

            smoothed = self._smooth(obj_id, speed_kmh)
            self.objects[obj_id]['speed_kmh'] = round(max(0.0, smoothed), 2)

            buf = self.speed_history.setdefault(obj_id, [])
            buf.append(smoothed)
            if len(buf) > 10:
                buf.pop(0)

    def _instantaneous_speed(self, history, fps):
        # Net displacement guard: if barely moved → stationary
        p0 = np.array(history[0][:2])
        p1 = np.array(history[-1][:2])
        if np.linalg.norm(p1 - p0) < 8.0:
            return 0.0

        speeds, weights = [], []
        for n, w in [(5, 0.2), (10, 0.5), (len(history), 0.3)]:
            if len(history) >= n:
                s = self._avg_speed_n(history, n, fps)
                if s is not None:
                    speeds.append(s)
                    weights.append(w)

        if not speeds:
            return 0.0

        result = sum(s * w for s, w in zip(speeds, weights)) / sum(weights)
        return min(result, self.realistic_speed_max)

    def _avg_speed_n(self, history, n, fps):
        """
        Total real-world distance / total time using per-instance MPP.
        ALL segments (including tiny movements) count toward both numerator
        and denominator, so genuine slow-moving vehicles aren't penalised.
        The 8px net-displacement guard in _instantaneous_speed already
        filters truly stationary vehicles.
        """
        pts = history[-n:]
        total_m, total_s = 0.0, 0.0
        for i in range(1, len(pts)):
            x1, y1, w1, h1 = pts[i - 1]
            x2, y2, w2, h2 = pts[i]

            dist_px = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            avg_w = (w1 + w2) / 2.0
            avg_h = (h1 + h2) / 2.0
            mpp   = self._get_mpp_from_bbox(avg_w, avg_h)

            total_m += dist_px * mpp   # all distances count
            total_s += 1.0 / fps       # all time counts

        return (total_m / total_s) * 3.6 if total_s > 0 else None

    def _smooth(self, obj_id, current_speed):
        prev  = self.objects[obj_id].get('speed_kmh', current_speed)
        alpha = self.smoothing_factor
        buf   = self.speed_history.get(obj_id, [])
        if len(buf) > 3:
            var = np.var(buf[-3:])
            if var > 100:
                alpha = 0.15
            elif var > 25:
                alpha = 0.25
        return ((alpha * current_speed) + ((1 - alpha) * prev)
                if prev > 0 else current_speed)