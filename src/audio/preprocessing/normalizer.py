"""Audio dynamic range normalization."""
import numpy as np

def normalize_decibels(mel_db: np.ndarray, min_db: float = -80.0) -> np.ndarray:
    """Clip and normalize mel spectrogram in decibels to [0, 1]."""
    clipped = np.clip(mel_db, min_db, 0.0)
    return (clipped - min_db) / (-min_db)
