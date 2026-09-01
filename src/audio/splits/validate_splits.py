"""Validate AEGIS generator-aware audio splits and fail loudly on leakage.

Validates speaker-based splits to ensure no speaker leakage across incompatible
evaluation boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from audio.splits.generator_split import SPLIT_ROLES, find_project_root, load_split_config
from audio.splits.leakage_checker import SplitRecord, check_splits

logger = logging.getLogger(__name__)


def load_split_csv(path: Path, split_role: str) -> list[SplitRecord]:
    """Load one split CSV into ``SplitRecord`` objects."""
    if not path.is_file():
        return []
    records: list[SplitRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
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
                    split_role=row.get("split_role", split_role),
                    dataset=row.get("dataset", "unknown"),
                    duration_sec=duration_sec,
                )
            )
    return records


def load_all_splits(output_dir: Path) -> dict[str, list[SplitRecord]]:
    """Load all split CSVs from ``data/processed/audio/splits/``."""
    return {
        role: load_split_csv(output_dir / f"{role}.csv", role)
        for role in SPLIT_ROLES
    }


def validate_split_files(project_root: Path, config_path: Path | None = None) -> int:
    """Validate on-disk split CSVs. Returns process exit code."""
    root = project_root.resolve()
    resolved_config = (config_path or root / "configs" / "audio_split.yaml").resolve()
    config = load_split_config(resolved_config)
    output_dir = (root / config.output_dir).resolve()

    splits = load_all_splits(output_dir)
    report = check_splits(
        splits,
        unseen_generators=config.unseen_generators,
        train_generators=config.split_generators.get("train", ()),
        min_minority_class_fraction=config.min_minority_class_fraction,
        incompatible_split_pairs=config.incompatible_split_pairs,
    )

    statistics_path = (root / config.statistics_path).resolve()
    if statistics_path.is_file():
        with statistics_path.open(encoding="utf-8") as handle:
            statistics = json.load(handle)
        statistics["leakage_summary"] = report.summary()
        with statistics_path.open("w", encoding="utf-8") as handle:
            json.dump(statistics, handle, indent=2)
            handle.write("\n")

    if report.passed:
        logger.info("Split validation passed.")
        for role in SPLIT_ROLES:
            logger.info("  %s: %d clips", role, len(splits.get(role, ())))
        
        # Report speaker counts
        for role in SPLIT_ROLES:
            speakers = {r.speaker_id for r in splits.get(role, ()) if r.speaker_id and r.speaker_id != "unknown"}
            logger.info("    └─ %d unique speakers", len(speakers))
        
        return 0

    logger.error("Split validation FAILED with %d violation(s):", len(report.violations))
    for violation in report.violations:
        logger.error("[%s] %s", violation.kind, violation.message)
        if violation.details:
            logger.error("  details: %s", violation.details)
    return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AEGIS audio experiment splits.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    project_root = (args.project_root or find_project_root()).resolve()
    return validate_split_files(project_root, config_path=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
