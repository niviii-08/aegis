"""Probability calibration with temperature scaling."""
import numpy as np

def apply_temperature_scaling(logit: float, temperature: float = 1.0) -> float:
    t = max(temperature, 1e-4)
    calibrated_logit = logit / t
    return 1.0 / (1.0 + np.exp(-calibrated_logit))
