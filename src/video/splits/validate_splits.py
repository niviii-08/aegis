"""Validate generated video splits against leakage policies."""

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from video.splits.generator_split import load_manifest_records, load_split_config
from video.splits.leakage_checker import check_splits

logger = logging.getLogger(__name__)

SPLIT_ROLES = ("train", "val", "test_seen", "test_unseen")

def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return current.parents[3]

def validate_splits(project_root: Path, config_path: Path | None = None) -> bool:
    root = project_root.resolve()
    resolved_config = (config_path or root / "configs" / "video_split.yaml").resolve()
    config = load_split_config(resolved_config)

    output_dir = (root / config.output_dir).resolve()
    grouped = {role: [] for role in SPLIT_ROLES}
    
    for role in SPLIT_ROLES:
        split_csv = output_dir / f"{role}.csv"
        if split_csv.is_file():
            grouped[role] = load_manifest_records(split_csv)
            
    report = check_splits(
        grouped,
        unseen_generators=config.unseen_generators,
        train_generators=config.split_generators.get("train", ()),
        min_minority_class_fraction=config.min_minority_class_fraction,
        incompatible_split_pairs=config.incompatible_split_pairs,
    )
    
    if report.passed:
        logger.info("All leakage checks passed successfully.")
        return True
        
    logger.error("Leakage validation failed:")
    for violation in report.violations:
        logger.error(f"- {violation.kind}: {violation.message}")
        
    return False

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project_root = find_project_root()
    
    passed = validate_splits(project_root, args.config)
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
