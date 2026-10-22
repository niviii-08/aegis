"""Tests for color space conversion."""
import numpy as np
from src.image.preprocessing.color_spaces import extract_hsv_saturation

def test_extract_hsv_saturation():
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    sat = extract_hsv_saturation(bgr)
    assert sat.shape == (50, 50)
