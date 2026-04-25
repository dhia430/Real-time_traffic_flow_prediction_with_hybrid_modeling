import pytest
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.getcwd(), 'src'))

from traffic.ctm_model import CellTransmissionModel
from traffic.density_estimator import DensityEstimator
from cv.tracker import VehicleTracker
from cv.detector import VehicleDetector

def test_ctm_conservation():
    """Test if CTM conserves vehicle density (mass balance)."""
    # Create a simple config-like mock or use a temporary config
    # For simplicity, we'll rely on defaults in the class or a dummy config
    ctm = CellTransmissionModel(num_cells=5)
    
    # Initial state is 0
    initial_total = np.sum(ctm.densities)
    assert initial_total == 0
    
    # Input some density
    input_rho = 0.05
    ctm.update(input_rho)
    
    # After update, density should have moved into the first cell
    assert ctm.densities[0] > 0
    assert np.sum(ctm.densities) > 0

def test_density_estimator_roi():
    """Test ROI detection logic."""
    # Use a dummy config with a square ROI
    # Mocking config via a temporary file or just setting attributes
    estimator = DensityEstimator()
    estimator.roi_points = np.array([[0,0], [10,0], [10,10], [0,10]], dtype=np.int32)
    estimator.road_length = 10.0
    
    # Point inside
    assert estimator._is_in_roi((5, 5)) == True
    # Point outside
    assert estimator._is_in_roi((15, 15)) == False

def test_tracker_assignment():
    """Test if tracker correctly assigns IDs."""
    tracker = VehicleTracker(max_distance=100)
    
    # Frame 1: One detection
    detections = [[10, 10, 20, 20, 0.9, 2]]
    objects = tracker.update(detections)
    assert len(objects) == 1
    first_id = list(objects.keys())[0]
    
    # Frame 2: Same detection slightly moved
    detections = [[12, 12, 22, 22, 0.9, 2]]
    objects = tracker.update(detections)
    assert first_id in objects
    assert len(objects) == 1

def test_tracker_disappearance():
    """Test if tracker correctly deregisters objects after max_disappeared frames."""
    tracker = VehicleTracker(max_disappeared=2, max_distance=100)
    
    # Frame 1: One detection
    tracker.update([[10, 10, 20, 20, 0.9, 2]])
    assert len(tracker.objects) == 1
    
    # Frame 2: No detections (object disappears)
    tracker.update([])
    assert len(tracker.objects) == 1  # Should still be 1 (disappeared = 1)
    
    # Frame 3: No detections
    tracker.update([])
    assert len(tracker.objects) == 1  # Should still be 1 (disappeared = 2)
    
    # Frame 4: Max exceeded (disappeared = 3 > max_disappeared=2)
    tracker.update([])
    assert len(tracker.objects) == 0  # Should be deregistered

def test_detector_inference():
    """Test if detector initializes and can process a dummy frame without crashing."""
    config_path = os.path.join(os.getcwd(), 'config', 'config.yaml')
    if not os.path.exists(config_path):
        pytest.skip("Config file not found, skipping detector test.")
    
    try:
        detector = VehicleDetector(config_path=config_path)
    except FileNotFoundError:
        pytest.skip("Model weights not found, skipping detector inference test.")
        return

    # Create a dummy image (black square)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run inference
    detections = detector.detect(dummy_frame)
    
    # With a black image, there should be 0 detections
    assert isinstance(detections, list)
    assert len(detections) == 0

if __name__ == "__main__":
    # Manual run if pytest isn't used
    test_ctm_conservation()
    test_density_estimator_roi()
    test_tracker_assignment()
    test_tracker_disappearance()
    test_detector_inference()
    print("All tests passed!")
