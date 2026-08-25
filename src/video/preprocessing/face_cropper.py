"""Face cropping utility."""
import cv2
import numpy as np
from dataclasses import dataclass
from video.preprocessing.face_detector import DetectedFace

@dataclass
class CropConfig:
    output_size: int = 224
    margin_factor: float = 0.25

class FaceCropper:
    """Crop and resize faces."""
    def __init__(self, crop_config: CropConfig = None):
        self.crop_config = crop_config or CropConfig()
        
    def process(self, image_rgb: np.ndarray, face: DetectedFace) -> np.ndarray | None:
        """Crop and resize face. Returns RGB cropped image or None."""
        h_img, w_img = image_rgb.shape[:2]
        margin_x = face.bbox_w * self.crop_config.margin_factor
        margin_y = face.bbox_h * self.crop_config.margin_factor
        
        x1 = max(0, int(face.bbox_x - margin_x))
        y1 = max(0, int(face.bbox_y - margin_y))
        x2 = min(w_img, int(face.bbox_x + face.bbox_w + margin_x))
        y2 = min(h_img, int(face.bbox_y + face.bbox_h + margin_y))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        cropped = image_rgb[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return None
            
        resized = cv2.resize(
            cropped, 
            (self.crop_config.output_size, self.crop_config.output_size), 
            interpolation=cv2.INTER_AREA
        )
        return resized
