"""Tests for frequency analysis."""
import numpy as np
from src.image.analysis.frequential import compute_fft_magnitude

def test_fft_magnitude():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    mag = compute_fft_magnitude(img)
    assert mag.shape == (64, 64)
