"""Tests for image filters."""
import numpy as np
from src.image.preprocessing.filters import compute_laplacian_variance

def test_laplacian_variance():
    flat = np.ones((100, 100), dtype=np.uint8) * 128
    assert compute_laplacian_variance(flat) == 0.0
