"""Tests for temperature calibration."""
from src.common.calibration import apply_temperature_scaling

def test_temperature_scaling():
    prob = apply_temperature_scaling(0.0, temperature=1.5)
    assert abs(prob - 0.5) < 1e-4
