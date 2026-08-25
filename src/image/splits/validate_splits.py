"""Validate AEGIS generator-aware image splits and fail loudly on leakage."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from image.data_audit import find_project_root
from image.splits.generator_split import SPLIT_ROLES, load_split_config
from image.splits.leakage_checker import SplitRecord, check_splits

logger = logging.getLogger(__name__)


def load_split_csv(path: Path, split_role: str) -> list[SplitRecord]:
    """Load one split CSV into ``SplitRecord`` objects."""
    if not path.is_file():
        return []
    records: list[SplitRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SplitRecord(
                    sample_id=row["sample_id"],
                    path=row["path"],
                    label=row["label"],
                    identity_key=row["identity_key"],
                    generator=row["generator"],
                    manipulation_method=row.get("manipulation_method", "unknown"),
                    original_source=row["original_source"],
                    source_image_key=row["source_image_key"],
                    file_hash=row["file_hash"],
                    split_role=row.get("split_role", split_role),
                    upstream_split=row.get("upstream_split", "unknown"),
                )
            )
    return records


def load_all_splits(output_dir: Path) -> dict[str, list[SplitRecord]]:
    """Load all split CSVs from ``data/processed/image/splits/``."""
    return {
        role: load_split_csv(output_dir / f"{role}.csv", role)
        for role in SPLIT_ROLES
    }


def validate_split_files(project_root: Path, config_path: Path | None = None) -> int:
    """Validate on-disk split CSVs. Returns process exit code."""
    root = project_root.resolve()
    resolved_config = (config_path or root / "configs" / "image_split.yaml").resolve()
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
            logger.info("  %s: %d samples", role, len(splits.get(role, ())))
        return 0

    logger.error("Split validation FAILED with %d violation(s):", len(report.violations))
    for violation in report.violations:
        logger.error("[%s] %s", violation.kind, violation.message)
        if violation.details:
            logger.error("  details: %s", violation.details)
    return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AEGIS image experiment splits.")
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
