"""Tests for face box smoothing."""
from src.video.preprocessing.face_tracker import smooth_bounding_boxes

def test_box_smoothing():
    boxes = [[10, 10, 50, 50], [12, 12, 52, 52]]
    smoothed = smooth_bounding_boxes(boxes)
    assert len(smoothed) == 2
