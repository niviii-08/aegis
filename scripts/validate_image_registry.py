"""
AEGIS Image Registry Validation Script
========================================
Validates the sample_registry.csv state and provides detailed statistics
for preprocessing readiness assessment.

Usage:
    python scripts/validate_image_registry.py
    python scripts/validate_image_registry.py --verbose
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def find_project_root() -> Path:
    """Locate AEGIS project root."""
    for parent in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise FileNotFoundError("Cannot locate AEGIS project root")


PROJECT_ROOT = find_project_root()
REGISTRY_PATH = PROJECT_ROOT / "data/processed/image/sample_registry.csv"
MANIFEST_PATH = PROJECT_ROOT / "data/processed/image/manifest.csv"
CROPS_DIR = PROJECT_ROOT / "data/processed/image/preprocessing/crops"
NORMALIZED_DIR = PROJECT_ROOT / "data/processed/image/preprocessing/normalized"
METADATA_PATH = PROJECT_ROOT / "data/processed/image/preprocessing/metadata.csv"


def validate_registry(verbose: bool = False) -> dict:
    """Run comprehensive registry validation."""
    print("=" * 80)
    print("AEGIS Image Registry Validation Report")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 80)

    # Load registry
    print("\n[1/8] Loading sample registry...")
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
    
    registry_df = pd.read_csv(REGISTRY_PATH, low_memory=False)
    total_rows = len(registry_df)
    print(f"  Total registry entries: {total_rows:,}")

    # Status distribution
    print("\n[2/8] Analyzing status distribution...")
    status_counts = registry_df["status"].value_counts()
    for status, count in status_counts.items():
        pct = (count / total_rows) * 100
        print(f"  {status:12s}: {count:>8,} ({pct:>5.2f}%)")

    # Split distribution
    print("\n[3/8] Analyzing split distribution...")
    split_counts = registry_df["split"].value_counts()
    for split_name, count in split_counts.items():
        pct = (count / total_rows) * 100
        print(f"  {split_name:12s}: {count:>8,} ({pct:>5.2f}%)")

    # Status by split
    print("\n[4/8] Analyzing status by split...")
    status_by_split = registry_df.groupby(["split", "status"]).size().unstack(fill_value=0)
    print(status_by_split.to_string())

    # Label distribution
    print("\n[5/8] Analyzing label distribution...")
    label_counts = registry_df["label"].value_counts()
    for label, count in label_counts.items():
        pct = (count / total_rows) * 100
        print(f"  {label:12s}: {count:>8,} ({pct:>5.2f}%)")

    # Generator distribution
    print("\n[6/8] Analyzing generator distribution...")
    gen_counts = registry_df["generator"].value_counts()
    for gen, count in gen_counts.items():
        pct = (count / total_rows) * 100
        print(f"  {gen:12s}: {count:>8,} ({pct:>5.2f}%)")

    # Processed samples analysis
    print("\n[7/8] Analyzing PROCESSED samples...")
    processed_df = registry_df[registry_df["status"] == "PROCESSED"]
    num_processed = len(processed_df)
    print(f"  Total PROCESSED: {num_processed:,}")
    
    if num_processed > 0:
        print(f"\n  PROCESSED by split:")
        proc_split = processed_df["split"].value_counts()
        for split_name, count in proc_split.items():
            pct = (count / num_processed) * 100
            print(f"    {split_name:12s}: {count:>8,} ({pct:>5.2f}%)")
        
        print(f"\n  PROCESSED by label:")
        proc_label = processed_df["label"].value_counts()
        for label, count in proc_label.items():
            pct = (count / num_processed) * 100
            print(f"    {label:12s}: {count:>8,} ({pct:>5.2f}%)")

        # Verify artifacts exist
        print(f"\n  Verifying artifacts for PROCESSED samples...")
        missing_crops = 0
        missing_normalized = 0
        for _, row in processed_df.iterrows():
            crop_path = PROJECT_ROOT / str(row.get("crop_path", "")) if pd.notna(row.get("crop_path")) and row.get("crop_path") else None
            norm_path = PROJECT_ROOT / str(row.get("processed_path", "")) if pd.notna(row.get("processed_path")) and row.get("processed_path") else None
            
            if crop_path and not crop_path.exists():
                missing_crops += 1
            if norm_path and not norm_path.exists():
                missing_normalized += 1
        
        print(f"    Missing crop files: {missing_crops:,}")
        print(f"    Missing normalized files: {missing_normalized:,}")

    # RAW samples ready for preprocessing
    print("\n[8/8] Analyzing RAW samples (ready for preprocessing)...")
    raw_df = registry_df[registry_df["status"] == "RAW"]
    num_raw = len(raw_df)
    print(f"  Total RAW (unprocessed): {num_raw:,}")
    
    if num_raw > 0:
        print(f"\n  RAW by split:")
        raw_split = raw_df["split"].value_counts()
        for split_name, count in raw_split.items():
            pct = (count / num_raw) * 100
            print(f"    {split_name:12s}: {count:>8,} ({pct:>5.2f}%)")
        
        print(f"\n  RAW by label:")
        raw_label = raw_df["label"].value_counts()
        for label, count in raw_label.items():
            pct = (count / num_raw) * 100
            print(f"    {label:12s}: {count:>8,} ({pct:>5.2f}%)")

    # Failed samples analysis
    print("\n[FAILED SAMPLES ANALYSIS]")
    failed_df = registry_df[registry_df["status"] == "FAILED"]
    num_failed = len(failed_df)
    print(f"  Total FAILED: {num_failed:,}")
    
    if num_failed > 0 and verbose:
        print(f"\n  Failure reasons:")
        if "failure_reason" in failed_df.columns:
            failure_reasons = failed_df["failure_reason"].value_counts()
            for reason, count in failure_reasons.items():
                print(f"    {reason}: {count}")

    # File counts
    print("\n[FILESYSTEM VERIFICATION]")
    if CROPS_DIR.exists():
        crop_files = list(CROPS_DIR.glob("*.jpg")) + list(CROPS_DIR.glob("*.png"))
        print(f"  Crop files on disk: {len(crop_files):,}")
    else:
        print(f"  Crop directory not found: {CROPS_DIR}")
    
    if NORMALIZED_DIR.exists():
        norm_files = list(NORMALIZED_DIR.glob("*.npy"))
        print(f"  Normalized files on disk: {len(norm_files):,}")
    else:
        print(f"  Normalized directory not found: {NORMALIZED_DIR}")
    
    if METADATA_PATH.exists():
        metadata_df = pd.read_csv(METADATA_PATH)
        print(f"  Metadata CSV rows: {len(metadata_df):,}")
    else:
        print(f"  Metadata CSV not found: {METADATA_PATH}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples in registry:        {total_rows:>10,}")
    print(f"PROCESSED (ready for training):   {num_processed:>10,} ({(num_processed/total_rows)*100:>5.2f}%)")
    print(f"RAW (needs preprocessing):        {num_raw:>10,} ({(num_raw/total_rows)*100:>5.2f}%)")
    print(f"FAILED (face detection failed):   {num_failed:>10,} ({(num_failed/total_rows)*100:>5.2f}%)")
    
    if num_raw > 0:
        print(f"\n⚠️  WARNING: {num_raw:,} samples have not been preprocessed yet!")
        print(f"   Run: python -m src.image.preprocessing.preprocess")
    else:
        print(f"\n✅ All samples have been preprocessed!")

    # Create summary dict for JSON export
    summary = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": int(total_rows),
        "status_distribution": {k: int(v) for k, v in status_counts.items()},
        "split_distribution": {k: int(v) for k, v in split_counts.items()},
        "label_distribution": {k: int(v) for k, v in label_counts.items()},
        "generator_distribution": {k: int(v) for k, v in gen_counts.items()},
        "processed_samples": int(num_processed),
        "raw_samples": int(num_raw),
        "failed_samples": int(num_failed),
        "preprocessing_complete": num_raw == 0,
    }

    return summary


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Validate AEGIS image sample registry state."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed failure reasons and individual sample checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write validation summary to JSON file.",
    )
    args = parser.parse_args()

    summary = validate_registry(verbose=args.verbose)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nValidation summary written to: {args.output}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
