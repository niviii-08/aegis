"""Video uniform frame sampling utilities."""
import numpy as np

def sample_frame_indices(total_frames: int, target_seq_len: int = 16) -> list[int]:
    if total_frames <= target_seq_len:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, target_seq_len, dtype=int).tolist()
