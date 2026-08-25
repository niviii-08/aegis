"""Face detection backend using facenet-pytorch."""
import logging
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DetectedFace:
    """One face detection result."""
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    confidence: float
    
    @property
    def area(self) -> int:
        return max(self.bbox_w, 0) * max(self.bbox_h, 0)


class FaceDetector:
    """MTCNN face detector backed by facenet-pytorch."""
    def __init__(self, min_confidence: float = 0.8):
        self.min_confidence = min_confidence
        # keep_all=True returns multiple faces if present
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
        
    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        try:
            boxes, probs = self.mtcnn.detect(image_rgb)
            if boxes is None:
                return []
                
            faces = []
            for box, prob in zip(boxes, probs):
                if prob is None or prob < self.min_confidence:
                    continue
                x1, y1, x2, y2 = box
                faces.append(DetectedFace(
                    bbox_x=int(x1),
                    bbox_y=int(y1),
                    bbox_w=int(x2 - x1),
                    bbox_h=int(y2 - y1),
                    confidence=float(prob)
                ))
            return faces
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            return []


def select_face(faces: Sequence[DetectedFace]) -> DetectedFace | None:
    """Select the largest face from multiple detections."""
    if not faces:
        return None
    return max(faces, key=lambda f: f.area)
