"""Unit tests for image dataset audit parsing and validation logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image.data_audit import (
    REAL_VS_FAKE_MEDIA_ROOT,
    classify_file,
    compute_leakage,
    infer_generator_bucket,
    infer_source_bucket,
    inspect_faceforensics_plus,
    load_csv_split,
    parse_csv_row,
    resolve_csv_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "image"


class TestClassification:
    def test_classify_image_extension(self) -> None:
        assert classify_file(Path("sample.JPG")) == "image"

    def test_classify_video_extension(self) -> None:
        assert classify_file(Path("clip.mp4")) == "video"

    def test_classify_python_code(self) -> None:
        assert classify_file(Path("module.py")) == "code"

    def test_classify_csv_metadata(self) -> None:
        assert classify_file(Path("train.csv")) == "metadata"


class TestCSVInference:
    def test_infer_ffhq_source(self) -> None:
        path = "/kaggle/input/flickrfaceshq-dataset-nvidia-part-1/images/02884.png"
        assert infer_source_bucket(path) == "ffhq_flickrfaceshq"

    def test_infer_stylegan_source(self) -> None:
        path = "/kaggle/input/1-million-fake-faces/1m_faces_08/FZV5C5L0AI.jpg"
        assert infer_source_bucket(path) == "1_million_fake_faces_stylegan"

    def test_infer_generator_real(self) -> None:
        assert infer_generator_bucket("/kaggle/input/ffhq/x.png", "real") == "none_authentic"

    def test_infer_generator_fake_stylegan(self) -> None:
        path = "/kaggle/input/1-million-fake-faces/1m_faces_08/FZV5C5L0AI.jpg"
        assert infer_generator_bucket(path, "fake") == "stylegan_1m_fake_faces"


class TestCSVRowParsing:
    def test_parse_valid_real_row(self) -> None:
        row = parse_csv_row(
            {
                "": "0",
                "original_path": "/kaggle/input/flickrfaceshq-dataset-nvidia-part-1/x.png",
                "id": "02884",
                "label": "1",
                "label_str": "real",
                "path": "train/real/02884.jpg",
            },
            "train",
        )
        assert row.sample_id == "02884"
        assert row.label == "1"
        assert row.generator_bucket == "none_authentic"

    def test_parse_valid_fake_row(self) -> None:
        row = parse_csv_row(
            {
                "": "0",
                "original_path": "/kaggle/input/1-million-fake-faces/1m_faces_08/FZV5C5L0AI.jpg",
                "id": "FZV5C5L0AI",
                "label": "0",
                "label_str": "fake",
                "path": "train/fake/FZV5C5L0AI.jpg",
            },
            "train",
        )
        assert row.label == "0"
        assert row.generator_bucket == "stylegan_1m_fake_faces"

    def test_parse_rejects_inconsistent_label(self) -> None:
        with pytest.raises(ValueError, match="Inconsistent label encoding"):
            parse_csv_row(
                {
                    "id": "1",
                    "label": "0",
                    "label_str": "real",
                    "path": "train/real/1.jpg",
                    "original_path": "",
                },
                "train",
            )


@pytest.mark.skipif(not (IMAGE_ROOT / "train.csv").is_file(), reason="train.csv missing")
class TestLocalCSVIntegration:
    def test_load_train_csv_split(self) -> None:
        stats = load_csv_split(IMAGE_ROOT / "train.csv", "train")
        assert stats.row_count == 100_000
        assert stats.label_str_counts["real"] == 50_000
        assert stats.label_str_counts["fake"] == 50_000
        assert stats.duplicate_ids == []

    def test_no_cross_split_id_leakage(self) -> None:
        train = load_csv_split(IMAGE_ROOT / "train.csv", "train")
        valid = load_csv_split(IMAGE_ROOT / "valid.csv", "valid")
        test = load_csv_split(IMAGE_ROOT / "test.csv", "test")

        train_ids = {row["id"] for row in json.loads(json.dumps(train.sample_rows))}
        # Use full reload via parse for leakage function instead
        from image.data_audit import load_split_rows_for_leakage

        rows_by_split = load_split_rows_for_leakage(IMAGE_ROOT)
        leakage = compute_leakage(rows_by_split)
        assert leakage.id_overlap["train_x_valid"] == 0
        assert leakage.id_overlap["train_x_test"] == 0
        assert leakage.id_overlap["valid_x_test"] == 0
        assert leakage.real_fake_id_collision_within_train == 0
        assert valid.row_count == 20_000
        assert test.row_count == 20_000
        assert train.row_count == 100_000

    def test_csv_paths_resolve_under_real_vs_fake(self) -> None:
        from image.data_audit import load_split_rows_for_leakage

        rows_by_split = load_split_rows_for_leakage(IMAGE_ROOT)
        resolution = resolve_csv_paths(IMAGE_ROOT, rows_by_split, spot_check_stride=5000)
        assert resolution["resolved_root"] is not None
        assert Path(resolution["resolved_root"]).name == "real-vs-fake"
        for split_stats in resolution["spot_checks"].values():
            assert split_stats["missing"] == 0


@pytest.mark.skipif(
    not (IMAGE_ROOT / "FaceForensics-master").is_dir(),
    reason="FaceForensics-master missing",
)
class TestFaceForensicsInspection:
    def test_ffpp_media_not_present_locally(self) -> None:
        result = inspect_faceforensics_plus(IMAGE_ROOT / "FaceForensics-master")
        assert result["repository_present"] is True
        assert result["ffpp_media_present_locally"] is False
        assert "split_json_files" in result
        assert result["split_json_files"]["train.json"]["pair_count"] == 360
