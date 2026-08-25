"""Face alignment, cropping, resizing, and normalization utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from image.preprocessing.face_detector import DetectedFace, FaceLandmarks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizationConfig:
    """ImageNet-style normalization parameters."""

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    scale_to_0_1_first: bool = True


@dataclass(frozen=True)
class CropConfig:
    """Parameters controlling face crop geometry."""

    margin_factor: float = 0.25
    output_size: int = 224
    align_faces: bool = True


@dataclass(frozen=True)
class CropResult:
    """Outputs from face cropping and normalization."""

    crop_rgb: np.ndarray
    normalized_chw: np.ndarray
    alignment_succeeded: bool
    selected_face_width: int
    selected_face_height: int
    error: str = ""


class FaceCropper:
    """Align, crop, resize, and normalize faces for model training."""

    def __init__(
        self,
        *,
        crop_config: CropConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ) -> None:
        self.crop_config = crop_config or CropConfig()
        self.normalization = normalization or NormalizationConfig()

    def process(self, image_rgb: np.ndarray, face: DetectedFace) -> CropResult:
        """Produce a standardized crop and normalized tensor from one detection."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return self._failure(image_rgb, face, "invalid_image_shape")

        working = image_rgb.copy()
        alignment_succeeded = False

        if self.crop_config.align_faces and face.landmarks is not None:
            aligned, alignment_succeeded = self._align_by_eyes(working, face.landmarks)
            if alignment_succeeded:
                working = aligned
            else:
                logger.debug("Eye alignment failed; continuing with unaligned crop.")

        cropped = self._crop_with_margin(working, face)
        if cropped.size == 0:
            return self._failure(image_rgb, face, "empty_crop")

        resized = cv2.resize(
            cropped,
            (self.crop_config.output_size, self.crop_config.output_size),
            interpolation=cv2.INTER_AREA if cropped.shape[0] > self.crop_config.output_size else cv2.INTER_LINEAR,
        )
        normalized = self.normalize(resized)

        return CropResult(
            crop_rgb=resized,
            normalized_chw=normalized,
            alignment_succeeded=alignment_succeeded,
            selected_face_width=face.bbox_w,
            selected_face_height=face.bbox_h,
        )

    def normalize(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Return float32 CHW tensor with configured normalization."""
        array = crop_rgb.astype(np.float32)
        if self.normalization.scale_to_0_1_first:
            array /= 255.0

        mean = np.array(self.normalization.mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(self.normalization.std, dtype=np.float32).reshape(3, 1, 1)
        chw = np.transpose(array, (2, 0, 1))
        return (chw - mean) / std

    def _align_by_eyes(
        self,
        image_rgb: np.ndarray,
        landmarks: FaceLandmarks,
    ) -> tuple[np.ndarray, bool]:
        left_eye = np.array(landmarks.left_eye, dtype=np.float32)
        right_eye = np.array(landmarks.right_eye, dtype=np.float32)

        if np.any(np.isnan(left_eye)) or np.any(np.isnan(right_eye)):
            return image_rgb, False

        delta = right_eye - left_eye
        if np.linalg.norm(delta) < 1e-3:
            return image_rgb, False

        angle_deg = float(np.degrees(np.arctan2(delta[1], delta[0])))
        eyes_center = ((left_eye + right_eye) / 2.0).tolist()
        rotation_matrix = cv2.getRotationMatrix2D(
            (float(eyes_center[0]), float(eyes_center[1])),
            angle_deg,
            1.0,
        )
        rotated = cv2.warpAffine(
            image_rgb,
            rotation_matrix,
            (image_rgb.shape[1], image_rgb.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, True

    def _crop_with_margin(self, image_rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        margin_x = int(round(face.bbox_w * self.crop_config.margin_factor))
        margin_y = int(round(face.bbox_h * self.crop_config.margin_factor))

        x0 = max(0, face.bbox_x - margin_x)
        y0 = max(0, face.bbox_y - margin_y)
        x1 = min(width, face.bbox_x + face.bbox_w + margin_x)
        y1 = min(height, face.bbox_y + face.bbox_h + margin_y)

        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0, 3), dtype=image_rgb.dtype)
        return image_rgb[y0:y1, x0:x1]

    def _failure(self, image_rgb: np.ndarray, face: DetectedFace, error: str) -> CropResult:
        size = self.crop_config.output_size
        empty_crop = np.zeros((size, size, 3), dtype=np.uint8)
        empty_norm = np.zeros((3, size, size), dtype=np.float32)
        return CropResult(
            crop_rgb=empty_crop,
            normalized_chw=empty_norm,
            alignment_succeeded=False,
            selected_face_width=face.bbox_w,
            selected_face_height=face.bbox_h,
            error=error,
        )
