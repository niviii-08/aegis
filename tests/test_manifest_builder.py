"""Tests for image manifest schema and builder."""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import pytest

from image.data.manifest_builder import (
    build_manifest,
    build_record_from_csv_row,
    deduplicate_records,
    detect_duplicate_content,
    read_jpeg_size,
    validate_image_file,
    write_manifest_outputs,
)
from image.data.manifest_schema import (
    MANIFEST_COLUMNS,
    UNKNOWN,
    ManifestRecord,
    infer_generator_from_provenance,
    infer_identity_id,
    infer_manipulation_method,
    make_sample_id,
    normalize_label,
    records_to_csv_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "image"
MEDIA_ROOT = IMAGE_ROOT / "real_vs_fake" / "real-vs-fake"


def _minimal_jpeg(width: int = 8, height: int = 6) -> bytes:
    """Build a tiny valid JPEG payload for tests."""
    sof = (
        b"\xff\xc0"
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03"
        + b"\x01\x11\x00"
        + b"\x02\x11\x01"
        + b"\x03\x11\x01"
    )
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + sof + b"\xff\xd9"


class TestManifestSchema:
    def test_normalize_label_from_label_str(self) -> None:
        assert normalize_label("1", "real") == "real"
        assert normalize_label("0", "fake") == "fake"

    def test_normalize_label_from_numeric_only(self) -> None:
        assert normalize_label("1", "") == "real"
        assert normalize_label("0", "") == "fake"

    def test_normalize_label_rejects_inconsistent(self) -> None:
        with pytest.raises(ValueError, match="Inconsistent label encoding"):
            normalize_label("0", "real")

    def test_make_sample_id(self) -> None:
        assert make_sample_id("real_vs_fake", "train", "31355") == "real_vs_fake:train:31355"

    def test_infer_generator_stylegan_from_path(self) -> None:
        path = "/kaggle/input/1-million-fake-faces/1m_faces_08/FZV5C5L0AI.jpg"
        assert infer_generator_from_provenance(path, "fake") == "stylegan"

    def test_infer_generator_real_is_unknown(self) -> None:
        path = "/kaggle/input/flickrfaceshq-dataset-nvidia-part-1/x.png"
        assert infer_generator_from_provenance(path, "real") == UNKNOWN

    def test_infer_manipulation_method_real_is_none(self) -> None:
        assert infer_manipulation_method("/kaggle/input/ffhq/x.png", "real") == "none"

    def test_infer_manipulation_method_fake_without_method_is_unknown(self) -> None:
        path = "/kaggle/input/1-million-fake-faces/1m_faces_08/FZV5C5L0AI.jpg"
        assert infer_manipulation_method(path, "fake") == UNKNOWN

    def test_identity_id_is_unknown_without_person_metadata(self) -> None:
        assert infer_identity_id("real", "02884", "/kaggle/input/ffhq/x.png") == UNKNOWN

    def test_manifest_record_csv_nullable_dimensions(self) -> None:
        record = ManifestRecord(
            sample_id="real_vs_fake:train:1",
            path="data/raw/image/real_vs_fake/real-vs-fake/train/real/1.jpg",
            dataset="real_vs_fake",
            modality="image",
            label="real",
            identity_id=UNKNOWN,
            generator=UNKNOWN,
            manipulation_method="none",
            original_source="/kaggle/input/ffhq/x.png",
            split="train",
            width=None,
            height=None,
            file_size=100,
            file_hash="abc",
            preprocessing_version=UNKNOWN,
        )
        row = record.to_csv_row()
        assert row["width"] == ""
        assert row["height"] == ""


class TestImageValidation:
    def test_read_jpeg_size(self) -> None:
        width, height = read_jpeg_size(_minimal_jpeg(32, 24))
        assert (width, height) == (32, 24)

    def test_validate_image_file_accepts_valid_jpeg(self, tmp_path: Path) -> None:
        image_path = tmp_path / "sample.jpg"
        image_path.write_bytes(_minimal_jpeg(16, 12))
        result = validate_image_file(image_path)
        assert result.valid is True
        assert result.width == 16
        assert result.height == 12
        assert result.file_hash

    def test_validate_image_file_rejects_empty_file(self, tmp_path: Path) -> None:
        image_path = tmp_path / "empty.jpg"
        image_path.write_bytes(b"")
        result = validate_image_file(image_path)
        assert result.valid is False
        assert result.error == "empty_file"

    def test_validate_image_file_rejects_invalid_jpeg(self, tmp_path: Path) -> None:
        image_path = tmp_path / "bad.jpg"
        image_path.write_bytes(b"not-a-jpeg")
        result = validate_image_file(image_path)
        assert result.valid is False


class TestManifestBuilderHelpers:
    def test_deduplicate_records_keeps_first_path(self) -> None:
        first = ManifestRecord(
            sample_id="real_vs_fake:train:a",
            path="data/a.jpg",
            dataset="real_vs_fake",
            modality="image",
            label="real",
            identity_id=UNKNOWN,
            generator=UNKNOWN,
            manipulation_method="none",
            original_source="src",
            split="train",
            width=10,
            height=10,
            file_size=10,
            file_hash="hash-a",
            preprocessing_version=UNKNOWN,
        )
        second = ManifestRecord(
            sample_id="real_vs_fake:train:b",
            path="data/a.jpg",
            dataset="real_vs_fake",
            modality="image",
            label="fake",
            identity_id=UNKNOWN,
            generator="stylegan",
            manipulation_method=UNKNOWN,
            original_source="src2",
            split="train",
            width=10,
            height=10,
            file_size=10,
            file_hash="hash-b",
            preprocessing_version=UNKNOWN,
        )
        records, summary = deduplicate_records([first, second])
        assert len(records) == 1
        assert summary.skipped_duplicate_path == 1

    def test_detect_duplicate_content(self) -> None:
        shared_hash = "deadbeef"
        records = [
            ManifestRecord(
                sample_id="real_vs_fake:train:1",
                path="data/one.jpg",
                dataset="real_vs_fake",
                modality="image",
                label="real",
                identity_id=UNKNOWN,
                generator=UNKNOWN,
                manipulation_method="none",
                original_source="src",
                split="train",
                width=10,
                height=10,
                file_size=10,
                file_hash=shared_hash,
                preprocessing_version=UNKNOWN,
            ),
            ManifestRecord(
                sample_id="real_vs_fake:train:2",
                path="data/two.jpg",
                dataset="real_vs_fake",
                modality="image",
                label="real",
                identity_id=UNKNOWN,
                generator=UNKNOWN,
                manipulation_method="none",
                original_source="src",
                split="train",
                width=10,
                height=10,
                file_size=10,
                file_hash=shared_hash,
                preprocessing_version=UNKNOWN,
            ),
        ]
        groups = detect_duplicate_content(records)
        assert shared_hash in groups
        assert len(groups[shared_hash]) == 2


@pytest.mark.skipif(not (MEDIA_ROOT / "train" / "real").is_dir(), reason="local media missing")
class TestLocalManifestIntegration:
    def test_build_record_from_existing_sample(self) -> None:
        sample = next((MEDIA_ROOT / "train" / "real").iterdir())
        raw_row = {
            "": "0",
            "original_path": f"/kaggle/input/flickrfaceshq-dataset-nvidia-part-1/x/{sample.stem}.png",
            "id": sample.stem,
            "label": "1",
            "label_str": "real",
            "path": f"train/real/{sample.name}",
        }
        record, status, _ = build_record_from_csv_row(
            split_name="train",
            raw_row=raw_row,
            media_root=MEDIA_ROOT,
            project_root=PROJECT_ROOT,
        )
        assert status == "ok"
        assert record is not None
        assert record.sample_id == f"real_vs_fake:train:{sample.stem}"
        assert record.label == "real"
        assert record.width is not None
        assert record.height is not None
        assert record.file_hash

    def test_build_manifest_full_corpus(self) -> None:
        records, summary = build_manifest(IMAGE_ROOT, PROJECT_ROOT)
        assert summary.records_written == 140_000
        assert summary.skipped_missing_file == 0
        assert summary.skipped_corrupted == 0
        assert summary.label_counts == {"fake": 70_000, "real": 70_000}
        assert summary.split_counts == {"test": 20_000, "train": 100_000, "valid": 20_000}
        assert not summary.duplicate_content_hash_groups
        assert records[0].sample_id < records[-1].sample_id

    def test_write_manifest_outputs_roundtrip(self, tmp_path: Path) -> None:
        records, summary = build_manifest(IMAGE_ROOT, PROJECT_ROOT)
        manifest_path = tmp_path / "manifest.csv"
        summary_path = tmp_path / "summary.json"
        write_manifest_outputs(
            records,
            summary,
            manifest_path=manifest_path,
            summary_path=summary_path,
        )

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == list(MANIFEST_COLUMNS)
            rows = list(reader)
        assert len(rows) == 140_000

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["records_written"] == 140_000

    def test_manifest_is_deterministic_across_reruns(self) -> None:
        first_records, _ = build_manifest(IMAGE_ROOT, PROJECT_ROOT)
        second_records, _ = build_manifest(IMAGE_ROOT, PROJECT_ROOT)
        first_rows = records_to_csv_rows(first_records)
        second_rows = records_to_csv_rows(second_records)
        assert first_rows == second_rows
