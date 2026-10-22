"""Temporal face bounding box smoothing."""
import numpy as np

def smooth_bounding_boxes(boxes: list[list[int]], alpha: float = 0.7) -> list[list[int]]:
    if not boxes:
        return []
    smoothed = [boxes[0]]
    for i in range(1, len(boxes)):
        curr = np.array(boxes[i])
        prev = np.array(smoothed[-1])
        new_box = (alpha * curr + (1.0 - alpha) * prev).astype(int).tolist()
        smoothed.append(new_box)
    return smoothed
