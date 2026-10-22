"""Canonical schema for the AEGIS audio data manifest.

The manifest is the single source of truth for audio experiments. Fields use
``unknown`` for missing categorical values and empty strings for absent numeric
metadata in CSV serialization.

ASVspoof Attack Codes:
  - A01–A06: ASVspoof 2019 LA seen attacks (training forgery generators)
  - A07–A19: ASVspoof 2021 LA attacks (additional generators)
  - bonafide: genuine human speech (label=real)
  - coqui: Coqui TTS generated clips (unseen generator)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

MANIFEST_VERSION = "1.0.0"

MANIFEST_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "file_path",
    "dataset",
    "modality",
    "label",
    "generator",
    "speaker_id",
    "duration_sec",
    "sample_rate",
    "file_size",
    "file_hash",
    "split",
    "preprocessing_version",
)

UNKNOWN = "unknown"
NONE = "none"

# ASVspoof 2019 LA attack types (seen generators for training)
ASVSPOOF_2019_ATTACKS = frozenset({
    "A01", "A02", "A03", "A04", "A05", "A06"
})

# ASVspoof 2021 LA attack types (extended set, mostly unseen)
ASVSPOOF_2021_ATTACKS = frozenset({
    "A07", "A08", "A09", "A10", "A11", "A12", "A13",
    "A14", "A15", "A16", "A17", "A18", "A19"
})

ALL_ASVSPOOF_ATTACKS = ASVSPOOF_2019_ATTACKS | ASVSPOOF_2021_ATTACKS


@dataclass(frozen=True)
class AudioManifestRecord:
    """One normalized manifest row for a usable audio sample."""

    clip_id: str
    file_path: str
    dataset: str
    modality: str
    label: str
    generator: str
    speaker_id: str
    duration_sec: float | None
    sample_rate: int | None
    file_size: int
    file_hash: str
    split: str
    preprocessing_version: str

    def to_csv_row(self) -> dict[str, str]:
        """Serialize to CSV-safe strings (null numerics become empty)."""
        row = asdict(self)
        for key in ("duration_sec", "sample_rate"):
            value = row[key]
            row[key] = "" if value is None else str(value)
        return {key: str(row[key]) for key in MANIFEST_COLUMNS}


def normalize_label(label_str: str) -> str:
    """Return canonical ``real`` / ``fake`` label from ASVspoof encodings.
    
    Args:
        label_str: Raw label string from protocol file ("bonafide" or "spoof")
        
    Returns:
        Canonical label: "real" or "fake"
        
    Raises:
        ValueError: If label_str is unrecognized
    """
    normalized = label_str.strip().lower()
    if normalized in {"bonafide", "genuine"}:
        return "real"
    if normalized in {"spoof", "fake"}:
        return "fake"
    raise ValueError(f"Unrecognized label encoding: {label_str!r}")


def normalize_generator(attack_type: str, label: str) -> str:
    """Map ASVspoof attack type to canonical generator name.
    
    Args:
        attack_type: Raw attack type from protocol file (e.g. "A01", "A19", "-")
        label: Canonical label ("real" or "fake")
        
    Returns:
        Generator name: "bonafide" for real, attack code for fake, or "unknown"
    """
    if label == "real":
        return "bonafide"
    
    attack = attack_type.strip().upper()
    if attack == "-" or not attack:
        return UNKNOWN
    
    if attack in ALL_ASVSPOOF_ATTACKS:
        return attack
    
    # Coqui TTS clips use "coqui" as generator
    if attack == "COQUI":
        return "coqui"
    
    return UNKNOWN


def infer_dataset_from_path(file_path: str) -> str:
    """Infer dataset name from file path structure."""
    normalized = file_path.replace("\\", "/").lower()
    if "asvspoof2019" in normalized:
        return "asvspoof2019_la"
    if "asvspoof2021" in normalized:
        return "asvspoof2021_la"
    if "coqui_tts" in normalized:
        return "coqui_tts_unseen"
    return UNKNOWN


def make_clip_id(dataset: str, split: str, source_id: str) -> str:
    """Build a deterministic, rerunnable clip identifier.
    
    Args:
        dataset: Dataset name (e.g. "asvspoof2019_la")
        split: Split name ("train", "val", "test_seen", "test_unseen")
        source_id: Original clip ID from protocol file
        
    Returns:
        Canonical clip_id string "dataset:split:source_id"
    """
    parts = (dataset.strip(), split.strip(), source_id.strip())
    if any(not part for part in parts):
        raise ValueError(
            f"Cannot build clip_id from dataset={dataset!r} split={split!r} id={source_id!r}"
        )
    return ":".join(parts)


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate a manifest record has required fields and allowed label values.
    
    Raises:
        ValueError: If record is missing required columns or has invalid values
    """
    missing = [column for column in MANIFEST_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Manifest record missing columns: {missing}")
    
    if record["label"] not in {"real", "fake"}:
        raise ValueError(f"Invalid label: {record['label']!r}")
    
    if not record["clip_id"]:
        raise ValueError("clip_id must not be empty")
    
    if not record["file_path"]:
        raise ValueError("file_path must not be empty")
    
    if record["modality"] != "audio":
        raise ValueError(f"Invalid modality for audio manifest: {record['modality']!r}")


def records_to_csv_rows(records: Sequence[AudioManifestRecord]) -> list[dict[str, str]]:
    """Convert manifest records to CSV row dicts in column order."""
    return [record.to_csv_row() for record in records]


def is_seen_generator(generator: str) -> bool:
    """Check if generator is part of the seen training set.
    
    Args:
        generator: Generator name (e.g. "A01", "A19", "bonafide")
        
    Returns:
        True if generator is bonafide or in ASVSPOOF_2019_ATTACKS
    """
    if generator == "bonafide":
        return True
    return generator in ASVSPOOF_2019_ATTACKS


def is_unseen_generator(generator: str) -> bool:
    """Check if generator is unseen (test_unseen split).
    
    Args:
        generator: Generator name (e.g. "A01", "A19", "coqui")
        
    Returns:
        True if generator is from ASVspoof 2021 LA or Coqui TTS
    """
    if generator == "coqui":
        return True
    return generator in ASVSPOOF_2021_ATTACKS
