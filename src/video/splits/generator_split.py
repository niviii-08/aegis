"""Build generator-aware, leakage-resistant video experiment splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from video.splits.leakage_checker import SplitRecord, check_splits

logger = logging.getLogger(__name__)

SPLIT_ROLES = ("train", "val", "test_seen", "test_unseen")

SPLIT_COLUMNS: tuple[str, ...] = (
    "video_id",
    "frame_path",
    "label",
    "identity_id",
    "generator",
    "original_source",
    "file_hash",
    "split_role",
)


@dataclass(frozen=True)
class SplitConfig:
    version: str
    manifest_path: Path
    output_dir: Path
    statistics_path: Path
    generator_taxonomy: dict[str, tuple[str, ...]]
    split_generators: dict[str, tuple[str, ...]]
    unseen_generators: tuple[str, ...]
    min_minority_class_fraction: float
    incompatible_split_pairs: tuple[tuple[str, str], ...]
    limitations: dict[str, Any]
    config_path: Path


@dataclass
class SplitStatistics:
    split_version: str
    build_timestamp: str
    config_path: str
    manifest_path: str
    total_manifest_records: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    label_counts_by_split: dict[str, dict[str, int]] = field(default_factory=dict)
    generator_counts_by_split: dict[str, dict[str, int]] = field(default_factory=dict)
    seen_generators: dict[str, list[str]] = field(default_factory=dict)
    unseen_generators: list[str] = field(default_factory=list)
    unseen_generator_samples_available: bool = False
    class_balance_by_split: dict[str, dict[str, float]] = field(default_factory=dict)
    limitations: dict[str, Any] = field(default_factory=dict)
    leakage_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def load_split_config(config_path: Path) -> SplitConfig:
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    taxonomy = raw.get("generator_taxonomy", {})
    split_generators = raw.get("split_generators", {})
    validation = raw.get("validation", {})

    unseen = tuple(taxonomy.get("unseen_forgery", ()))
    incompatible_pairs = tuple(tuple(pair) for pair in validation.get("incompatible_split_pairs", []))

    return SplitConfig(
        version=str(raw.get("version", "unknown")),
        manifest_path=Path(raw["manifest_path"]),
        output_dir=Path(raw["output_dir"]),
        statistics_path=Path(raw["statistics_path"]),
        generator_taxonomy={k: tuple(v) for k, v in taxonomy.items()},
        split_generators={k: tuple(v) for k, v in split_generators.items()},
        unseen_generators=unseen,
        min_minority_class_fraction=float(validation.get("min_minority_class_fraction", 0.10)),
        incompatible_split_pairs=incompatible_pairs,
        limitations=dict(raw.get("limitations", {})),
        config_path=config_path.resolve(),
    )


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return current.parents[3]


def load_manifest_records(manifest_path: Path) -> list[SplitRecord]:
    records: list[SplitRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SplitRecord(
                    video_id=row["video_id"],
                    frame_path=row["frame_path"],
                    label=row["label"],
                    identity_id=row["identity_id"],
                    generator=row["generator"],
                    original_source=row["original_source"],
                    file_hash=row["file_hash"],
                    split_role="unassigned",
                )
            )
    records.sort(key=lambda x: x.video_id)
    return records


def deterministic_split(video_id: str, train_pct=0.7, val_pct=0.15) -> str:
    """Return train, val, or test_seen deterministically based on hashing video_id."""
    digest = hashlib.md5(video_id.encode('utf-8')).hexdigest()
    val = int(digest[:8], 16) / 0xFFFFFFFF
    if val < train_pct:
        return "train"
    elif val < train_pct + val_pct:
        return "val"
    else:
        return "test_seen"


def assign_split_roles(records: Sequence[SplitRecord], config: SplitConfig) -> list[SplitRecord]:
    assigned: list[SplitRecord] = []
    unseen_generators = set(config.unseen_generators)

    for record in records:
        if record.generator in unseen_generators:
            split_role = "test_unseen"
        else:
            split_role = deterministic_split(record.video_id)

        allowed_generators = set(config.split_generators.get(split_role, ()))
        if record.generator not in allowed_generators:
            logger.warning(f"Generator {record.generator} not expected in {split_role}.")

        assigned.append(
            SplitRecord(
                video_id=record.video_id,
                frame_path=record.frame_path,
                label=record.label,
                identity_id=record.identity_id,
                generator=record.generator,
                original_source=record.original_source,
                file_hash=record.file_hash,
                split_role=split_role,
            )
        )
    return assigned


def write_split_csv(path: Path, records: Sequence[SplitRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SPLIT_COLUMNS))
        writer.writeheader()
        for record in sorted(records, key=lambda x: x.video_id):
            writer.writerow({
                "video_id": record.video_id,
                "frame_path": record.frame_path,
                "label": record.label,
                "identity_id": record.identity_id,
                "generator": record.generator,
                "original_source": record.original_source,
                "file_hash": record.file_hash,
                "split_role": record.split_role,
            })
    temp_path.replace(path)


def compute_class_balance(label_counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(label_counts.values())
    if total == 0:
        return {"real_fraction": 0.0, "fake_fraction": 0.0, "minority_fraction": 0.0}
    real_frac = label_counts.get("real", 0) / total
    fake_frac = label_counts.get("fake", 0) / total
    return {
        "real_fraction": round(real_frac, 6),
        "fake_fraction": round(fake_frac, 6),
        "minority_fraction": round(min(real_frac, fake_frac), 6),
    }


def build_generator_splits(project_root: Path, config_path: Path | None = None, validate: bool = True) -> SplitStatistics:
    root = project_root.resolve()
    resolved_config = (config_path or root / "configs" / "video_split.yaml").resolve()
    config = load_split_config(resolved_config)

    manifest_path = (root / config.manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = load_manifest_records(manifest_path)
    assigned = assign_split_roles(records, config)

    grouped = {role: [] for role in SPLIT_ROLES}
    for record in assigned:
        grouped[record.split_role].append(record)

    output_dir = (root / config.output_dir).resolve()
    for role in SPLIT_ROLES:
        write_split_csv(output_dir / f"{role}.csv", grouped[role])

    leakage_report = check_splits(
        grouped,
        unseen_generators=config.unseen_generators,
        train_generators=config.split_generators.get("train", ()),
        min_minority_class_fraction=config.min_minority_class_fraction,
        incompatible_split_pairs=config.incompatible_split_pairs,
    )
    if validate and not leakage_report.passed:
        raise RuntimeError(
            "Split construction produced leakage or balance violations:\n"
            + "\n".join(v.message for v in leakage_report.violations)
        )

    label_counts_by_split = {role: dict(Counter(r.label for r in grouped[role])) for role in SPLIT_ROLES}
    stats = SplitStatistics(
        split_version=config.version,
        build_timestamp=datetime.now(timezone.utc).isoformat(),
        config_path=str(config.config_path),
        manifest_path=str(manifest_path),
        total_manifest_records=len(records),
        split_counts={role: len(grouped[role]) for role in SPLIT_ROLES},
        label_counts_by_split=label_counts_by_split,
        generator_counts_by_split={role: dict(Counter(r.generator for r in grouped[role])) for role in SPLIT_ROLES},
        seen_generators={role: list(config.split_generators.get(role, ())) for role in SPLIT_ROLES},
        unseen_generators=list(config.unseen_generators),
        unseen_generator_samples_available=any(r.generator in config.unseen_generators for r in records),
        class_balance_by_split={role: compute_class_balance(label_counts_by_split[role]) for role in SPLIT_ROLES},
        limitations=config.limitations,
        leakage_summary=leakage_report.summary(),
        notes=["Splits derived dynamically based on hashing the video_id."],
    )

    stats_path = (root / config.statistics_path).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build AEGIS video splits.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project_root = find_project_root()
    
    try:
        stats = build_generator_splits(project_root, config_path=args.config, validate=not args.skip_validation)
        logger.info("Split build complete.")
        logger.info("Counts: %s", stats.split_counts)
    except Exception as e:
        logger.error(f"Failed to build splits: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
