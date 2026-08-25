"""Tests for image preprocessing utilities and pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from image.preprocessing.face_cropper import CropConfig, FaceCropper, NormalizationConfig
from image.preprocessing.face_detector import DetectedFace, FaceLandmarks, select_face
from image.preprocessing.preprocess import (
    METADATA_COLUMNS,
    build_metadata_row,
    load_preprocess_config,
    safe_output_basename,
    should_skip_sample,
    validate_image,
    write_failures_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "image_preprocessing.yaml"


class TestFaceSelection:
    def test_select_largest_face(self) -> None:
        faces = [
            DetectedFace(bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10, confidence=0.95),
            DetectedFace(bbox_x=0, bbox_y=0, bbox_w=30, bbox_h=30, confidence=0.80),
        ]
        selected = select_face(faces, policy="largest", image_width=100, image_height=100)
        assert selected is not None
        assert selected.bbox_w == 30

    def test_select_highest_confidence(self) -> None:
        faces = [
            DetectedFace(bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10, confidence=0.95),
            DetectedFace(bbox_x=0, bbox_y=0, bbox_w=30, bbox_h=30, confidence=0.80),
        ]
        selected = select_face(faces, policy="highest_confidence", image_width=100, image_height=100)
        assert selected is not None
        assert selected.confidence == 0.95


class TestFaceCropper:
    def test_normalize_output_shape(self) -> None:
        cropper = FaceCropper(
            crop_config=CropConfig(output_size=224),
            normalization=NormalizationConfig(),
        )
        face = DetectedFace(
            bbox_x=20,
            bbox_y=20,
            bbox_w=80,
            bbox_h=80,
            confidence=0.99,
            landmarks=FaceLandmarks(
                left_eye=(40.0, 50.0),
                right_eye=(80.0, 50.0),
                nose=(60.0, 70.0),
                mouth_left=(45.0, 90.0),
                mouth_right=(75.0, 90.0),
            ),
        )
        image = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
        result = cropper.process(image, face)
        assert result.crop_rgb.shape == (224, 224, 3)
        assert result.normalized_chw.shape == (3, 224, 224)
        assert result.normalized_chw.dtype == np.float32


class TestPreprocessHelpers:
    def test_load_preprocess_config(self) -> None:
        config = load_preprocess_config(CONFIG_PATH, PROJECT_ROOT)
        assert config.version == "1.0.0"
        assert config.detector_name == "mtcnn"
        assert config.output_size == 224

    def test_safe_output_basename(self) -> None:
        assert safe_output_basename("real_vs_fake:train:00001") == "real_vs_fake__train__00001"

    def test_should_skip_successful_sample(self) -> None:
        config = load_preprocess_config(CONFIG_PATH, PROJECT_ROOT)
        existing = {
            "sample:a": {
                "sample_id": "sample:a",
                "preprocessing_version": config.version,
                "status": "success",
                "processed_crop_path": "data/processed/image/preprocessing/crops/sample.jpg",
                "processed_normalized_path": "data/processed/image/preprocessing/normalized/sample.npy",
            }
        }
        config.crops_dir.mkdir(parents=True, exist_ok=True)
        config.normalized_dir.mkdir(parents=True, exist_ok=True)
        crop_file = PROJECT_ROOT / existing["sample:a"]["processed_crop_path"]
        norm_file = PROJECT_ROOT / existing["sample:a"]["processed_normalized_path"]
        crop_file.parent.mkdir(parents=True, exist_ok=True)
        norm_file.parent.mkdir(parents=True, exist_ok=True)
        crop_file.write_bytes(b"fake")
        norm_file.write_bytes(b"fake")
        assert should_skip_sample("sample:a", config, existing) is True
        assert should_skip_sample("sample:b", config, existing) is False

    def test_validate_real_manifest_image(self) -> None:
        config = load_preprocess_config(CONFIG_PATH, PROJECT_ROOT)
        manifest_path = PROJECT_ROOT / "data" / "processed" / "image" / "manifest.csv"
        if not manifest_path.is_file():
            pytest.skip("manifest.csv not available locally")
        import csv

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        image_path = PROJECT_ROOT / row["path"]
        if not image_path.is_file():
            pytest.skip("sample image missing locally")
        outcome = validate_image(image_path, config)
        assert outcome.valid is True
        assert outcome.image_rgb is not None

    def test_build_metadata_row_contains_required_fields(self) -> None:
        config = load_preprocess_config(CONFIG_PATH, PROJECT_ROOT)
        row = build_metadata_row(
            sample_id="real_vs_fake:train:1",
            original_path="data/raw/image/x.jpg",
            config=config,
            status="no_face",
            error_message="no_detectable_face",
        )
        assert set(row.keys()) == set(METADATA_COLUMNS)
        assert row["detector"] == "mtcnn"
        assert row["preprocessing_version"] == config.version

    def test_write_failures_csv(self, tmp_path: Path) -> None:
        config = load_preprocess_config(CONFIG_PATH, PROJECT_ROOT)
        rows = [
            build_metadata_row(
                sample_id="a",
                original_path="p/a.jpg",
                config=config,
                status="success",
            ),
            build_metadata_row(
                sample_id="b",
                original_path="p/b.jpg",
                config=config,
                status="no_face",
                error_message="no_detectable_face",
            ),
        ]
        out = tmp_path / "failures.csv"
        write_failures_csv(rows, out)
        text = out.read_text(encoding="utf-8")
        assert "sample_id" in text
        assert "no_face" in text
        assert "success" not in text.splitlines()[1]
