"""Spatial filtering utilities for image forensics."""
import cv2
import numpy as np

def compute_laplacian_variance(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
