"""Tests for video flicker analyzer."""
import numpy as np
from src.video.analysis.flicker import compute_interframe_difference

def test_flicker():
    f = np.zeros((4, 20, 20, 3), dtype=np.float32)
    diff = compute_interframe_difference(f)
    assert diff == 0.0
