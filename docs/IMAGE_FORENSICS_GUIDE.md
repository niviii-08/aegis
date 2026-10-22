# Image Forensics Guide

AEGIS combines deep CNN features with domain heuristics:
1. **Laplacian Variance**: Measures high-frequency gradient density at 224x224.
2. **HSV Paper Detection**: Identifies low-saturation, high-value art paper textures.
3. **Hough Line Transform**: Detects horizontal notebook ruled lines.
