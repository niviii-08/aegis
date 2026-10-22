"""Video temporal flicker analysis."""
import numpy as np

def compute_interframe_difference(frames: np.ndarray) -> float:
    diffs = np.abs(frames[1:] - frames[:-1])
    return float(np.mean(diffs))
