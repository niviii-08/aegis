"""Build generator-aware, leakage-resistant image experiment splits.

Reads the canonical manifest and ``configs/image_split.yaml``, assigns each sample
to ``train``, ``val``, ``test_seen``, or ``test_unseen``, and writes split CSVs
plus ``reports/split_statistics.json``.

When unseen-generator media is unavailable locally, ``test_unseen`` is written
with headers only and limitations are recorded explicitly rather than faking
held-out generators.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from image.data_audit import find_project_root
from image.splits.leakage_checker import SplitRecord, check_splits

logger = logging.getLogger(__name__)

SPLIT_ROLES = ("train", "val", "test_seen", "test_unseen")

SPLIT_COLUMNS: tuple[str, ...] = (
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

AUTHENTIC_GENERATOR = "ffhq_authentic"
STYLEGAN_GENERATOR = "stylegan"


@dataclass(frozen=True)
class SplitConfig:
    """Parsed split policy from ``configs/image_split.yaml``."""

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
    """Load and normalize the YAML split policy."""
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
        generator_taxonomy={key: tuple(values) for key, values in taxonomy.items()},
        split_generators={role: tuple(values) for role, values in split_generators.items()},
        source_split_mapping={str(k): str(v) for k, v in raw.get("source_split_mapping", {}).items()},
        unseen_generators=unseen,
        min_minority_class_fraction=float(validation.get("min_minority_class_fraction", 0.10)),
        incompatible_split_pairs=incompatible_pairs,
        limitations=dict(raw.get("limitations", {})),
        config_path=config_path.resolve(),
    )


def normalize_generator(label: str, generator: str, original_source: str) -> str:
    """Map manifest generator fields to canonical config identifiers."""
    normalized_source = original_source.replace("\\", "/").lower()
    if label == "real":
        if "flickrfaceshq" in normalized_source or "ffhq" in normalized_source:
            return AUTHENTIC_GENERATOR
        return AUTHENTIC_GENERATOR
    if generator == STYLEGAN_GENERATOR:
        return STYLEGAN_GENERATOR
    if "1-million-fake-faces" in normalized_source or "stylegan" in normalized_source:
        return STYLEGAN_GENERATOR
    if "deepfakes" in normalized_source:
        return "deepfakes"
    if "face2face" in normalized_source:
        return "face2face"
    if "faceswap" in normalized_source:
        return "faceswap"
    if "neuraltextures" in normalized_source:
        return "neuraltextures"
    if "faceshifter" in normalized_source:
        return "faceshifter"
    if "deepfake" in normalized_source and "detection" in normalized_source:
        return "deepfake_detection"
    return generator if generator and generator != "unknown" else "unknown_forgery"


def extract_source_id(sample_id: str) -> str:
    """Return the upstream sample identifier embedded in ``sample_id``."""
    parts = sample_id.split(":")
    if len(parts) < 3 or not parts[-1]:
        raise ValueError(f"Cannot parse source id from sample_id={sample_id!r}")
    return parts[-1]


def derive_identity_key(label: str, source_id: str) -> str:
    """Build a proxy identity key from label and upstream sample id.

    This is not verified person identity. It prevents the same upstream sample
    identifier from appearing across incompatible evaluation splits.
    """
    if label == "real":
        return f"ffhq_source:{source_id}"
    if label == "fake":
        return f"stylegan_face:{source_id}"
    return f"unknown_source:{source_id}"


def derive_source_image_key(original_source: str) -> str:
    """Normalize upstream provenance paths for cross-split source-image checks."""
    cleaned = original_source.strip().replace("\\", "/")
    if not cleaned or cleaned == "unknown":
        return "unknown"
    return cleaned.lower()


def load_manifest_records(manifest_path: Path) -> list[SplitRecord]:
    """Load manifest rows and enrich them with split-assignment metadata."""
    records: list[SplitRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = extract_source_id(row["sample_id"])
            label = row["label"]
            generator = normalize_generator(label, row["generator"], row["original_source"])
            records.append(
                SplitRecord(
                    sample_id=row["sample_id"],
                    path=row["path"],
                    label=label,
                    identity_key=derive_identity_key(label, source_id),
                    generator=generator,
                    manipulation_method=row["manipulation_method"],
                    original_source=row["original_source"],
                    source_image_key=derive_source_image_key(row["original_source"]),
                    file_hash=row["file_hash"],
                    split_role="unassigned",
                    upstream_split=row["split"],
                )
            )
    records.sort(key=lambda item: item.sample_id)
    return records


def assign_split_roles(records: Sequence[SplitRecord], config: SplitConfig) -> list[SplitRecord]:
    """Assign each record to an evaluation split role."""
    assigned: list[SplitRecord] = []
    unseen_generators = set(config.unseen_generators)

    for record in records:
        if record.generator in unseen_generators:
            split_role = "test_unseen"
        elif record.upstream_split in config.source_split_mapping:
            split_role = config.source_split_mapping[record.upstream_split]
        else:
            raise ValueError(
                f"Manifest upstream split {record.upstream_split!r} is not mapped in config "
                f"for sample_id={record.sample_id}"
            )

        allowed_generators = set(config.split_generators.get(split_role, ()))
        if record.generator not in allowed_generators:
            raise ValueError(
                f"Generator {record.generator!r} is not allowed in split role {split_role!r} "
                f"for sample_id={record.sample_id}"
            )

        assigned.append(
            SplitRecord(
                sample_id=record.sample_id,
                path=record.path,
                label=record.label,
                identity_key=record.identity_key,
                generator=record.generator,
                manipulation_method=record.manipulation_method,
                original_source=record.original_source,
                source_image_key=record.source_image_key,
                file_hash=record.file_hash,
                split_role=split_role,
                upstream_split=record.upstream_split,
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
        "sample_id": record.sample_id,
        "path": record.path,
        "label": record.label,
        "identity_key": record.identity_key,
        "generator": record.generator,
        "manipulation_method": record.manipulation_method,
        "original_source": record.original_source,
        "source_image_key": record.source_image_key,
        "file_hash": record.file_hash,
        "split_role": record.split_role,
        "upstream_split": record.upstream_split,
    }


def write_split_csv(path: Path, records: Sequence[SplitRecord]) -> None:
    """Write one split CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SPLIT_COLUMNS))
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.sample_id):
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
    unseen_available = any(record.generator in config.unseen_generators for record in records)

    notes = [
        "Split roles are assigned deterministically from configs/image_split.yaml.",
        "Unseen forgery generators are fixed in config and never sampled at runtime.",
    ]
    if not unseen_available:
        notes.append(
            "test_unseen contains zero samples because no local media matches unseen forgery generators."
        )

    return SplitStatistics(
        split_version=config.version,
        build_timestamp=datetime.now(timezone.utc).isoformat(),
        config_path=str(config.config_path),
        manifest_path=str(manifest_path.resolve()),
        total_manifest_records=len(records),
        split_counts={role: len(grouped.get(role, ())) for role in SPLIT_ROLES},
        label_counts_by_split=label_counts_by_split,
        generator_counts_by_split=generator_counts_by_split,
        seen_generators={role: list(config.split_generators.get(role, ())) for role in SPLIT_ROLES},
        unseen_generators=list(config.unseen_generators),
        unseen_generator_samples_available=unseen_available,
        class_balance_by_split={
            role: compute_class_balance(label_counts_by_split[role]) for role in SPLIT_ROLES
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
    resolved_config = (config_path or root / "configs" / "image_split.yaml").resolve()
    config = load_split_config(resolved_config)

    manifest_path = (root / config.manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = load_manifest_records(manifest_path)
    assigned = assign_split_roles(records, config)
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
    parser = argparse.ArgumentParser(description="Build AEGIS generator-aware image splits.")
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
        help="Path to image_split.yaml (defaults to configs/image_split.yaml).",
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
    config = load_split_config((args.config or project_root / "configs" / "image_split.yaml").resolve())
    logger.info("Statistics: %s", (project_root / config.statistics_path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
