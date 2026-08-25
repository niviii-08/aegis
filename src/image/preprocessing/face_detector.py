"""Face detection backends for the AEGIS image preprocessing pipeline.

Uses MTCNN (TensorFlow) by default because it is compatible with the project's
existing TensorFlow stack and provides bounding boxes, confidence scores, and
facial landmarks required for alignment.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_DETECTORS = frozenset({"mtcnn"})


@dataclass(frozen=True)
class FaceLandmarks:
    """Canonical facial landmark coordinates in pixel space."""

    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]

    @property
    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "nose": self.nose,
            "mouth_left": self.mouth_left,
            "mouth_right": self.mouth_right,
        }


@dataclass(frozen=True)
class DetectedFace:
    """One face detection result."""

    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    confidence: float
    landmarks: FaceLandmarks | None = None

    @property
    def area(self) -> int:
        return max(self.bbox_w, 0) * max(self.bbox_h, 0)

    @property
    def center(self) -> tuple[float, float]:
        return (self.bbox_x + self.bbox_w / 2.0, self.bbox_y + self.bbox_h / 2.0)

    def to_metadata(self) -> dict[str, str | float | int]:
        payload: dict[str, str | float | int] = {
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_w": self.bbox_w,
            "bbox_h": self.bbox_h,
            "detection_confidence": round(self.confidence, 6),
        }
        if self.landmarks is not None:
            for name, (x_coord, y_coord) in self.landmarks.as_dict.items():
                payload[f"landmark_{name}_x"] = round(x_coord, 2)
                payload[f"landmark_{name}_y"] = round(y_coord, 2)
        return payload


@dataclass
class DetectionResult:
    """All faces detected in one image."""

    faces: list[DetectedFace] = field(default_factory=list)
    detector: str = ""
    error: str = ""

    @property
    def face_count(self) -> int:
        return len(self.faces)


class FaceDetector(ABC):
    """Abstract face detector interface."""

    name: str

    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> DetectionResult:
        """Detect faces in an RGB uint8 image array (H, W, 3)."""


class MTCNNFaceDetector(FaceDetector):
    """MTCNN face detector backed by the ``mtcnn`` package (TensorFlow)."""

    name = "mtcnn"

    def __init__(
        self,
        *,
        min_face_size: int = 20,
        min_confidence: float = 0.0,
    ) -> None:
        self.min_face_size = min_face_size
        self.min_confidence = min_confidence
        self._detector: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._detector is None:
            logger.info("Loading MTCNN face detector (first use may take a moment)...")
            from mtcnn import MTCNN

            self._detector = MTCNN()
            logger.info("MTCNN face detector ready.")
        return self._detector

    def detect(self, image_rgb: np.ndarray) -> DetectionResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return DetectionResult(faces=[], detector=self.name, error="invalid_image_shape")

        height, width = image_rgb.shape[:2]
        if height <= 0 or width <= 0:
            return DetectionResult(faces=[], detector=self.name, error="invalid_image_dimensions")

        try:
            raw_faces = self._ensure_loaded().detect_faces(image_rgb)
        except Exception as exc:
            logger.debug("MTCNN detection failed: %s", exc)
            return DetectionResult(faces=[], detector=self.name, error=f"detector_error:{exc}")

        faces: list[DetectedFace] = []
        for raw in raw_faces or []:
            parsed = self._parse_face(raw, image_width=width, image_height=height)
            if parsed is not None:
                faces.append(parsed)

        faces.sort(key=lambda face: face.confidence, reverse=True)
        return DetectionResult(faces=faces, detector=self.name)

    def _parse_face(
        self,
        raw: Mapping[str, Any],
        *,
        image_width: int,
        image_height: int,
    ) -> DetectedFace | None:
        confidence = float(raw.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return None

        box = raw.get("box")
        if not box or len(box) != 4:
            return None

        x, y, w, h = (int(round(float(value))) for value in box)
        if w < self.min_face_size or h < self.min_face_size:
            return None

        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        w = max(1, min(w, image_width - x))
        h = max(1, min(h, image_height - y))

        keypoints = raw.get("keypoints") or {}
        landmarks = self._parse_landmarks(keypoints) if keypoints else None

        return DetectedFace(
            bbox_x=x,
            bbox_y=y,
            bbox_w=w,
            bbox_h=h,
            confidence=confidence,
            landmarks=landmarks,
        )

    @staticmethod
    def _parse_landmarks(keypoints: Mapping[str, Any]) -> FaceLandmarks | None:
        required = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
        parsed: dict[str, tuple[float, float]] = {}
        for name in required:
            point = keypoints.get(name)
            if not point or len(point) != 2:
                return None
            parsed[name] = (float(point[0]), float(point[1]))
        return FaceLandmarks(**parsed)


def create_face_detector(
    name: str,
    *,
    min_face_size: int = 20,
    min_confidence: float = 0.0,
) -> FaceDetector:
    """Instantiate a supported face detector by name."""
    normalized = name.strip().lower()
    if normalized not in SUPPORTED_DETECTORS:
        supported = ", ".join(sorted(SUPPORTED_DETECTORS))
        raise ValueError(f"Unsupported detector {name!r}. Supported: {supported}")
    if normalized == "mtcnn":
        return MTCNNFaceDetector(min_face_size=min_face_size, min_confidence=min_confidence)
    raise ValueError(f"Detector {name!r} is listed but not implemented.")


def select_face(
    faces: Sequence[DetectedFace],
    *,
    policy: str,
    image_width: int,
    image_height: int,
) -> DetectedFace | None:
    """Select one face from multiple detections according to policy."""
    if not faces:
        return None

    normalized_policy = policy.strip().lower()
    if normalized_policy == "largest":
        return max(faces, key=lambda face: face.area)
    if normalized_policy == "highest_confidence":
        return max(faces, key=lambda face: face.confidence)
    if normalized_policy == "center":
        image_center = (image_width / 2.0, image_height / 2.0)

        def center_distance(face: DetectedFace) -> float:
            cx, cy = face.center
            return (cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2

        return min(faces, key=center_distance)

    raise ValueError(
        f"Unsupported multi_face_policy {policy!r}. "
        "Expected one of: largest, highest_confidence, center."
    )
