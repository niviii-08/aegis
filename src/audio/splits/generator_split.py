"""Build generator-aware, speaker-based audio experiment splits.

Reads the canonical manifest and ``configs/audio_split.yaml``, assigns each clip
to ``train``, ``val``, ``test_seen``, or ``test_unseen`` with strict speaker-level
constraints, and writes split CSVs plus ``reports/audio/split_statistics.json``.

CRITICAL: All clips from the same speaker_id are assigned to the same split to
prevent speaker leakage across evaluation boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from audio.data.manifest_schema import MANIFEST_COLUMNS
from audio.splits.leakage_checker import SplitRecord, check_splits

logger = logging.getLogger(__name__)

SPLIT_ROLES = ("train", "val", "test_seen", "test_unseen")

SPLIT_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "file_path",
    "label",
    "speaker_id",
    "generator",
    "dataset",
    "file_hash",
    "split_role",
    "upstream_split",
    "duration_sec",
)


@dataclass(frozen=True)
class SplitConfig:
    """Parsed split policy from ``configs/audio_split.yaml``."""

    version: str
    manifest_path: Path
    output_dir: Path
    statistics_path: Path
    generator_taxonomy: dict[str, tuple[str, ...]]
    split_generators: dict[str, tuple[str, ...]]
    source_split_mapping: dict[str, str]
    unseen_generators: tuple[str, ...]
    min_minority_class_fraction: float
    incompatible_split_pairs: tuple[tuple[str, str], ...]
    speaker_strategy: str
    min_clips_per_speaker: int
    target_ratios: dict[str, float]
    limitations: dict[str, Any]
    config_path: Path


@dataclass
class SplitStatistics:
    """Machine-readable summary emitted after split construction."""

    split_version: str
    build_timestamp: str
    config_path: str
    manifest_path: str
    total_manifest_records: int = 0
    total_speakers: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    speaker_counts_by_split: dict[str, int] = field(default_factory=dict)
    label_counts_by_split: dict[str, dict[str, int]] = field(default_factory=dict)
    generator_counts_by_split: dict[str, dict[str, int]] = field(default_factory=dict)
    seen_generators: dict[str, list[str]] = field(default_factory=dict)
    unseen_generators: list[str] = field(default_factory=list)
    class_balance_by_split: dict[str, dict[str, float]] = field(default_factory=dict)
    average_duration_by_split: dict[str, float] = field(default_factory=dict)
    limitations: dict[str, Any] = field(default_factory=dict)
    leakage_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def find_project_root() -> Path:
    """Locate the AEGIS project root by searching for src/ directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "src").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root (no src/ directory found)")


def load_split_config(config_path: Path) -> SplitConfig:
    """Load and normalize the YAML split policy."""
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    taxonomy = raw.get("generator_taxonomy", {})
    split_generators = raw.get("split_generators", {})
    validation = raw.get("validation", {})
    speaker_split = raw.get("speaker_split", {})

    unseen = tuple(taxonomy.get("unseen_forgery", ()))
    incompatible_pairs = tuple(tuple(pair) for pair in validation.get("incompatible_split_pairs", []))

    return SplitConfig(
        version=str(raw.get("version", "unknown")),
        manifest_path=Path(raw["manifest_path"]),
        output_dir=Path(raw["output_dir"]),
        statistics_path=Path(raw["statistics_path"]),
        generator_taxonomy={key: tuple(values) for key, values in taxonomy.items()},
        split_generators={role: tuple(values) for role, values in split_generators.items()},
        source_split_mapping={str(k): str(v) for k, v in raw.get("source_split_mapping", {}).items()},
        unseen_generators=unseen,
        min_minority_class_fraction=float(validation.get("min_minority_class_fraction", 0.10)),
        incompatible_split_pairs=incompatible_pairs,
        speaker_strategy=str(speaker_split.get("strategy", "by_speaker")),
        min_clips_per_speaker=int(speaker_split.get("min_clips_per_speaker", 1)),
        target_ratios=dict(speaker_split.get("target_ratios", {"train": 0.70, "val": 0.15, "test_seen": 0.15})),
        limitations=dict(raw.get("limitations", {})),
        config_path=config_path.resolve(),
    )


def load_manifest_records(manifest_path: Path) -> list[SplitRecord]:
    """Load manifest rows and enrich them with split-assignment metadata."""
    records: list[SplitRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            duration_sec = None
            if row.get("duration_sec"):
                try:
                    duration_sec = float(row["duration_sec"])
                except (ValueError, TypeError):
                    pass
            
            records.append(
                SplitRecord(
                    clip_id=row["clip_id"],
                    file_path=row["file_path"],
                    label=row["label"],
                    speaker_id=row.get("speaker_id", "unknown"),
                    generator=row["generator"],
                    file_hash=row["file_hash"],
                    split_role="unassigned",
                    dataset=row.get("dataset", "unknown"),
                    duration_sec=duration_sec,
                )
            )
    records.sort(key=lambda item: item.clip_id)
    return records


def group_clips_by_speaker(
    records: Sequence[SplitRecord],
) -> dict[str, list[SplitRecord]]:
    """Group clips by speaker_id."""
    by_speaker: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in records:
        by_speaker[record.speaker_id].append(record)
    return dict(by_speaker)


def assign_split_roles_speaker_based(
    records: Sequence[SplitRecord],
    config: SplitConfig,
) -> list[SplitRecord]:
    unseen_generators = set(config.unseen_generators)
    assigned: list[SplitRecord] = []
    
    unseen_records = [r for r in records if r.generator in unseen_generators]
    seen_records = [r for r in records if r.generator not in unseen_generators]
    
    unseen_speakers = {r.speaker_id for r in unseen_records if r.speaker_id and r.speaker_id != "unknown"}
    
    for record in records:
        if record.generator in unseen_generators or record.speaker_id in unseen_speakers:
            assigned.append(
                SplitRecord(
                    clip_id=record.clip_id,
                    file_path=record.file_path,
                    label=record.label,
                    speaker_id=record.speaker_id,
                    generator=record.generator,
                    file_hash=record.file_hash,
                    split_role="test_unseen",
                    dataset=record.dataset,
                    duration_sec=record.duration_sec,
                )
            )
            
    pure_seen_records = [r for r in seen_records if r.speaker_id not in unseen_speakers]
    by_speaker = group_clips_by_speaker(pure_seen_records)
    
    filtered_speakers = {
        spk: clips
        for spk, clips in by_speaker.items()
        if len(clips) >= config.min_clips_per_speaker
    }
    
    if not filtered_speakers:
        logger.warning("No speakers meet minimum clip threshold")
        return assigned
    
    speaker_ids = sorted(filtered_speakers.keys())
    
    train_speakers, val_speakers, test_seen_speakers = [], [], []
    import hashlib
    for speaker_id in speaker_ids:
        speaker_hash = int(hashlib.md5(speaker_id.encode('utf-8')).hexdigest(), 16)
        roll = (speaker_hash % 100) / 100.0
        
        if roll < config.target_ratios["train"]:
            train_speakers.append(speaker_id)
        elif roll < config.target_ratios["train"] + config.target_ratios["val"]:
            val_speakers.append(speaker_id)
        else:
            test_seen_speakers.append(speaker_id)
            
    speaker_to_split = {}
    for spk in train_speakers: speaker_to_split[spk] = "train"
    for spk in val_speakers: speaker_to_split[spk] = "val"
    for spk in test_seen_speakers: speaker_to_split[spk] = "test_seen"
    
    for record in pure_seen_records:
        split_role = speaker_to_split.get(record.speaker_id, "train")
        assigned.append(
            SplitRecord(
                clip_id=record.clip_id,
                file_path=record.file_path,
                label=record.label,
                speaker_id=record.speaker_id,
                generator=record.generator,
                file_hash=record.file_hash,
                split_role=split_role,
                dataset=record.dataset,
                duration_sec=record.duration_sec,
            )
        )
    return assigned


def group_records_by_split(records: Sequence[SplitRecord]) -> dict[str, list[SplitRecord]]:
    """Bucket records by ``split_role``."""
    grouped: dict[str, list[SplitRecord]] = {role: [] for role in SPLIT_ROLES}
    for record in records:
        grouped.setdefault(record.split_role, []).append(record)
    return grouped


def record_to_csv_row(record: SplitRecord) -> dict[str, str]:
    """Serialize one split record to CSV."""
    return {
        "clip_id": record.clip_id,
        "file_path": record.file_path,
        "label": record.label,
        "speaker_id": record.speaker_id,
        "generator": record.generator,
        "dataset": record.dataset,
        "file_hash": record.file_hash,
        "split_role": record.split_role,
        "upstream_split": "",  # Not used for audio
        "duration_sec": f"{record.duration_sec:.3f}" if record.duration_sec is not None else "",
    }


def write_split_csv(path: Path, records: Sequence[SplitRecord]) -> None:
    """Write one split CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SPLIT_COLUMNS))
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.clip_id):
            writer.writerow(record_to_csv_row(record))
    temp_path.replace(path)


def compute_class_balance(label_counts: Mapping[str, int]) -> dict[str, float]:
    """Return real/fake fractions for one split."""
    total = sum(label_counts.values())
    if total == 0:
        return {"real_fraction": 0.0, "fake_fraction": 0.0, "minority_fraction": 0.0}
    real_fraction = label_counts.get("real", 0) / total
    fake_fraction = label_counts.get("fake", 0) / total
    return {
        "real_fraction": round(real_fraction, 6),
        "fake_fraction": round(fake_fraction, 6),
        "minority_fraction": round(min(real_fraction, fake_fraction), 6),
    }


def compute_average_duration(records: Sequence[SplitRecord]) -> float:
    """Compute average duration for a split."""
    durations = [r.duration_sec for r in records if r.duration_sec is not None]
    if not durations:
        return 0.0
    return sum(durations) / len(durations)


def count_unique_speakers(records: Sequence[SplitRecord]) -> int:
    """Count unique speakers in a split."""
    speakers = {r.speaker_id for r in records if r.speaker_id and r.speaker_id != "unknown"}
    return len(speakers)


def build_statistics(
    records: Sequence[SplitRecord],
    grouped: Mapping[str, Sequence[SplitRecord]],
    config: SplitConfig,
    manifest_path: Path,
    leakage_summary: Mapping[str, Any],
) -> SplitStatistics:
    """Aggregate split statistics for reporting."""
    label_counts_by_split = {
        role: dict(Counter(record.label for record in grouped.get(role, ())))
        for role in SPLIT_ROLES
    }
    generator_counts_by_split = {
        role: dict(Counter(record.generator for record in grouped.get(role, ())))
        for role in SPLIT_ROLES
    }
    
    all_speakers = {r.speaker_id for r in records if r.speaker_id and r.speaker_id != "unknown"}

    notes = [
        "Split roles are assigned deterministically from configs/audio_split.yaml.",
        "All clips from the same speaker_id are in the same split (speaker-based splitting).",
        "Unseen forgery generators (A07–A19, coqui) are fixed and assigned to test_unseen exclusively.",
    ]

    return SplitStatistics(
        split_version=config.version,
        build_timestamp=datetime.now(timezone.utc).isoformat(),
        config_path=str(config.config_path),
        manifest_path=str(manifest_path.resolve()),
        total_manifest_records=len(records),
        total_speakers=len(all_speakers),
        split_counts={role: len(grouped.get(role, ())) for role in SPLIT_ROLES},
        speaker_counts_by_split={
            role: count_unique_speakers(grouped.get(role, ()))
            for role in SPLIT_ROLES
        },
        label_counts_by_split=label_counts_by_split,
        generator_counts_by_split=generator_counts_by_split,
        seen_generators={role: list(config.split_generators.get(role, ())) for role in SPLIT_ROLES},
        unseen_generators=list(config.unseen_generators),
        class_balance_by_split={
            role: compute_class_balance(label_counts_by_split[role]) for role in SPLIT_ROLES
        },
        average_duration_by_split={
            role: compute_average_duration(grouped.get(role, ()))
            for role in SPLIT_ROLES
        },
        limitations=config.limitations,
        leakage_summary=dict(leakage_summary),
        notes=notes,
    )


def write_statistics(path: Path, statistics: SplitStatistics) -> None:
    """Write split statistics JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(statistics), handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def build_generator_splits(
    project_root: Path,
    config_path: Path | None = None,
    *,
    validate: bool = True,
) -> SplitStatistics:
    """Construct split CSVs and statistics from the manifest and config."""
    root = project_root.resolve()
    resolved_config = (config_path or root / "configs" / "audio_split.yaml").resolve()
    config = load_split_config(resolved_config)

    manifest_path = (root / config.manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = load_manifest_records(manifest_path)
    assigned = assign_split_roles_speaker_based(records, config)
    grouped = group_records_by_split(assigned)

    output_dir = (root / config.output_dir).resolve()
    for role in SPLIT_ROLES:
        write_split_csv(output_dir / f"{role}.csv", grouped.get(role, ()))

    grouped_for_validation = {role: grouped.get(role, ()) for role in SPLIT_ROLES}
    leakage_report = check_splits(
        grouped_for_validation,
        unseen_generators=config.unseen_generators,
        train_generators=config.split_generators.get("train", ()),
        min_minority_class_fraction=config.min_minority_class_fraction,
        incompatible_split_pairs=config.incompatible_split_pairs,
    )
    
    if validate and not leakage_report.passed:
        raise RuntimeError(
            "Split construction produced leakage or balance violations:\n"
            + "\n".join(violation.message for violation in leakage_report.violations)
        )

    statistics = build_statistics(
        assigned,
        grouped,
        config,
        manifest_path,
        leakage_summary=leakage_report.summary(),
    )
    write_statistics((root / config.statistics_path).resolve(), statistics)
    return statistics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AEGIS generator-aware audio splits with speaker-based splitting.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="AEGIS project root (defaults to auto-discovery).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to audio_split.yaml (defaults to configs/audio_split.yaml).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Write outputs even if leakage checks fail (not recommended).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    project_root = (args.project_root or find_project_root()).resolve()
    statistics = build_generator_splits(
        project_root,
        config_path=args.config,
        validate=not args.skip_validation,
    )
    logger.info("Split build complete.")
    logger.info("Counts: %s", statistics.split_counts)
    logger.info("Speakers: %s", statistics.speaker_counts_by_split)
    config = load_split_config((args.config or project_root / "configs" / "audio_split.yaml").resolve())
    logger.info("Statistics: %s", (project_root / config.statistics_path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
