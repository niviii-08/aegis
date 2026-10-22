"""2D Fourier spectrum analysis for synthetic artifacts."""
import numpy as np

def compute_fft_magnitude(img_gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    return 20 * np.log(np.abs(fshift) + 1e-8)
