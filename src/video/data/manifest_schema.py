"""Canonical schema for the AEGIS video data manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

MANIFEST_VERSION = "1.0.0"

# Note: We use 'frame_path' as the column name to strictly match the requested
# schema, though it will point to the raw .mp4 video file prior to V2 extraction.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "video_id",
    "frame_path",
    "label",
    "generator",
    "identity_id",
    "original_source",
    "file_hash",
    "split",
)

UNKNOWN = "unknown"
NONE = "none"


@dataclass(frozen=True)
class ManifestRecord:
    """One normalized manifest row for a usable video sample."""

    video_id: str
    frame_path: str
    label: str
    generator: str
    identity_id: str
    original_source: str
    file_hash: str
    split: str

    def to_csv_row(self) -> dict[str, str]:
        """Serialize to CSV-safe strings."""
        row = asdict(self)
        return {key: str(row[key]) for key in MANIFEST_COLUMNS}


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate a manifest record has required fields and allowed label values."""
    missing = [column for column in MANIFEST_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Manifest record missing columns: {missing}")
    if record["label"] not in {"real", "fake"}:
        raise ValueError(f"Invalid label: {record['label']!r}")
    if not record["video_id"]:
        raise ValueError("video_id must not be empty")
    if not record["frame_path"]:
        raise ValueError("frame_path must not be empty")


def records_to_csv_rows(records: Sequence[ManifestRecord]) -> list[dict[str, str]]:
    """Convert manifest records to CSV row dicts in column order."""
    return [record.to_csv_row() for record in records]
