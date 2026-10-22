"""Color space extraction utilities."""
import cv2
import numpy as np

def extract_hsv_saturation(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1]
