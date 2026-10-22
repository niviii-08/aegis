"""Canonical schema for the AEGIS image data manifest.

The manifest is the single source of truth for image experiments. Fields use
``unknown`` for missing categorical values and empty strings for absent numeric
metadata in CSV serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

MANIFEST_VERSION = "1.0.0"

MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "path",
    "dataset",
    "modality",
    "label",
    "identity_id",
    "generator",
    "manipulation_method",
    "original_source",
    "split",
    "width",
    "height",
    "file_size",
    "file_hash",
    "preprocessing_version",
)

UNKNOWN = "unknown"
NONE = "none"


@dataclass(frozen=True)
class ManifestRecord:
    """One normalized manifest row for a usable image sample."""

    sample_id: str
    path: str
    dataset: str
    modality: str
    label: str
    identity_id: str
    generator: str
    manipulation_method: str
    original_source: str
    split: str
    width: int | None
    height: int | None
    file_size: int
    file_hash: str
    preprocessing_version: str

    def to_csv_row(self) -> dict[str, str]:
        """Serialize to CSV-safe strings (null numerics become empty)."""
        row = asdict(self)
        for key in ("width", "height"):
            value = row[key]
            row[key] = "" if value is None else str(value)
        return {key: str(row[key]) for key in MANIFEST_COLUMNS}


def normalize_label(label: str, label_str: str) -> str:
    """Return canonical ``real`` / ``fake`` label from raw CSV encodings."""
    normalized = label_str.strip().lower()
    if normalized in {"real", "fake"}:
        if normalized == "real" and label.strip() not in {"", "1"}:
            raise ValueError(f"Inconsistent label encoding: label_str=real label={label!r}")
        if normalized == "fake" and label.strip() not in {"", "0"}:
            raise ValueError(f"Inconsistent label encoding: label_str=fake label={label!r}")
        return normalized
    numeric = label.strip()
    if numeric == "1":
        return "real"
    if numeric == "0":
        return "fake"
    raise ValueError(f"Unrecognized label encoding: label={label!r} label_str={label_str!r}")


def make_sample_id(dataset: str, split: str, source_id: str) -> str:
    """Build a deterministic, rerunnable sample identifier."""
    parts = (dataset.strip(), split.strip(), source_id.strip())
    if any(not part for part in parts):
        raise ValueError(f"Cannot build sample_id from dataset={dataset!r} split={split!r} id={source_id!r}")
    return ":".join(parts)


def infer_generator_from_provenance(original_path: str, label: str) -> str:
    """Infer generator only when explicitly supported by ``original_path``."""
    normalized = original_path.replace("\\", "/").lower()
    if label == "real":
        return UNKNOWN
    if "1-million-fake-faces" in normalized or "1m_faces" in normalized:
        return "stylegan"
    if "stylegan" in normalized:
        return "stylegan"
    if "deepfake" in normalized or "deepfakes" in normalized:
        return UNKNOWN
    if label == "fake":
        return UNKNOWN
    return UNKNOWN


def infer_manipulation_method(original_path: str, label: str) -> str:
    """Infer manipulation method only when absence or method is explicit."""
    if label == "real":
        return NONE
    normalized = original_path.replace("\\", "/").lower()
    if "deepfakes" in normalized:
        return "deepfakes"
    if "face2face" in normalized:
        return "face2face"
    if "faceswap" in normalized:
        return "faceswap"
    if "neuraltextures" in normalized:
        return "neuraltextures"
    if "faceshifter" in normalized:
        return "faceshifter"
    if label == "fake":
        return UNKNOWN
    return UNKNOWN


def infer_original_source(original_path: str) -> str:
    """Preserve upstream provenance path when available."""
    cleaned = original_path.strip()
    return cleaned if cleaned else UNKNOWN


def infer_identity_id(label: str, source_id: str, original_path: str) -> str:
    """Return identity when explicitly available; otherwise ``unknown``.

    The real-vs-fake CSV ``id`` is a per-image identifier (FFHQ index or
    synthetic face code), not a verified cross-sample person identity.
    """
    _ = (label, source_id, original_path)
    return UNKNOWN


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate a manifest record has required fields and allowed label values."""
    missing = [column for column in MANIFEST_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Manifest record missing columns: {missing}")
    if record["label"] not in {"real", "fake"}:
        raise ValueError(f"Invalid label: {record['label']!r}")
    if not record["sample_id"]:
        raise ValueError("sample_id must not be empty")
    if not record["path"]:
        raise ValueError("path must not be empty")


def records_to_csv_rows(records: Sequence[ManifestRecord]) -> list[dict[str, str]]:
    """Convert manifest records to CSV row dicts in column order."""
    return [record.to_csv_row() for record in records]
