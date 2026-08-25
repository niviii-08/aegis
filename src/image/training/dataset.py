"""Load face-crop samples exclusively from split manifests and preprocessing metadata."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from image.preprocessing.face_cropper import NormalizationConfig

logger = logging.getLogger(__name__)

InputSource = Literal["normalized_npy", "crop_jpeg"]
SPLIT_COLUMNS = (
    "sample_id",
    "path",
    "label",
    "identity_key",
    "generator",
    "manipulation_method",
    "original_source",
    "source_image_key",
    "file_hash",
    "split_role",
    "upstream_split",
)

METADATA_COLUMNS = (
    "sample_id",
    "original_path",
    "processed_crop_path",
    "processed_normalized_path",
    "preprocessing_version",
    "status",
)

LABEL_TO_INT = {"real": 0, "fake": 1}


@dataclass(frozen=True)
class SampleRecord:
    """One training/evaluation sample resolved from manifest joins."""

    sample_id: str
    label: int
    generator: str
    split_role: str
    crop_path: Path | None
    normalized_path: Path | None
    preprocessing_version: str


def load_split_csv(split_csv: Path) -> list[dict[str, str]]:
    """Load rows from a generated split manifest CSV."""
    if not split_csv.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_csv}")

    rows: list[dict[str, str]] = []
    with split_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {split_csv}")
        for row in reader:
            rows.append({col: row.get(col, "") for col in SPLIT_COLUMNS})
    rows.sort(key=lambda item: item["sample_id"])
    return rows


def load_preprocessing_index(
    metadata_path: Path,
    *,
    preprocessing_version: str,
    success_status: str = "success",
) -> dict[str, dict[str, str]]:
    """Index successful preprocessing rows by ``sample_id``."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Preprocessing metadata not found: {metadata_path}")

    index: dict[str, dict[str, str]] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != success_status:
                continue
            if row.get("preprocessing_version") != preprocessing_version:
                continue
            sample_id = row.get("sample_id", "")
            if sample_id:
                index[sample_id] = row
    return index


def resolve_samples(
    split_csv: Path,
    metadata_path: Path,
    project_root: Path,
    *,
    preprocessing_version: str,
    input_source: InputSource,
) -> list[SampleRecord]:
    """Join split manifest rows with preprocessing metadata; never scan directories."""
    split_rows = load_split_csv(split_csv)
    preprocessing_index = load_preprocessing_index(
        metadata_path,
        preprocessing_version=preprocessing_version,
    )

    samples: list[SampleRecord] = []
    missing_metadata = 0
    missing_files = 0

    for row in split_rows:
        sample_id = row["sample_id"]
        meta = preprocessing_index.get(sample_id)
        if meta is None:
            missing_metadata += 1
            continue

        crop_rel = meta.get("processed_crop_path", "").strip()
        norm_rel = meta.get("processed_normalized_path", "").strip()
        crop_path = (project_root / crop_rel).resolve() if crop_rel else None
        normalized_path = (project_root / norm_rel).resolve() if norm_rel else None

        if input_source == "normalized_npy":
            if normalized_path is None or not normalized_path.is_file():
                missing_files += 1
                continue
        else:
            if crop_path is None or not crop_path.is_file():
                missing_files += 1
                continue

        label_str = row["label"]
        if label_str not in LABEL_TO_INT:
            raise ValueError(f"Unknown label {label_str!r} for sample_id={sample_id}")

        samples.append(
            SampleRecord(
                sample_id=sample_id,
                label=LABEL_TO_INT[label_str],
                generator=row["generator"],
                split_role=row["split_role"],
                crop_path=crop_path,
                normalized_path=normalized_path,
                preprocessing_version=meta.get("preprocessing_version", preprocessing_version),
            )
        )

    logger.info(
        "Resolved %d samples from %s (%d missing metadata, %d missing files)",
        len(samples),
        split_csv.name,
        missing_metadata,
        missing_files,
    )
    return samples


def measure_class_balance(samples: Sequence[SampleRecord]) -> dict[str, float | int]:
    """Return class counts and minority fraction for weighting decisions."""
    fake_count = sum(1 for sample in samples if sample.label == 1)
    real_count = sum(1 for sample in samples if sample.label == 0)
    total = fake_count + real_count
    if total == 0:
        return {
            "total": 0,
            "fake_count": 0,
            "real_count": 0,
            "fake_fraction": 0.0,
            "real_fraction": 0.0,
            "minority_fraction": 0.0,
        }
    fake_fraction = fake_count / total
    real_fraction = real_count / total
    return {
        "total": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_fraction": round(fake_fraction, 6),
        "real_fraction": round(real_fraction, 6),
        "minority_fraction": round(min(fake_fraction, real_fraction), 6),
    }


class FaceCropDataset(Dataset):
    """PyTorch dataset backed by split manifests and preprocessing outputs."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        *,
        input_source: InputSource = "normalized_npy",
        normalization: NormalizationConfig | None = None,
    ) -> None:
        self.samples = list(samples)
        self.input_source = input_source
        self.normalization = normalization or NormalizationConfig()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        if self.input_source == "normalized_npy":
            assert sample.normalized_path is not None
            array = np.load(sample.normalized_path)
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            tensor = torch.from_numpy(array)
        else:
            assert sample.crop_path is not None
            tensor = self._load_jpeg_crop(sample.crop_path)

        label = torch.tensor(sample.label, dtype=torch.float32)
        return tensor, label, sample.sample_id

    def _load_jpeg_crop(self, crop_path: Path) -> torch.Tensor:
        import cv2

        bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Failed to read crop image: {crop_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.normalization.scale_to_0_1_first:
            rgb /= 255.0
        mean = np.array(self.normalization.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.normalization.std, dtype=np.float32).reshape(1, 1, 3)
        normalized = (rgb - mean) / std
        chw = np.transpose(normalized, (2, 0, 1))
        return torch.from_numpy(chw.copy())
