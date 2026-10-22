"""Tests for audio normalizer."""
import numpy as np
from src.audio.preprocessing.normalizer import normalize_decibels

def test_normalize_decibels():
    mel = np.array([[-80.0, -40.0, 0.0]])
    norm = normalize_decibels(mel)
    assert np.isclose(norm[0, 0], 0.0)
    assert np.isclose(norm[0, 1], 0.5)
    assert np.isclose(norm[0, 2], 1.0)
