"""Quick preprocessing progress check (single snapshot, no file locking).

Usage:
    python scripts/check_progress.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    """Find AEGIS project root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "data").exists() and (current / "configs").exists():
            return current
        current = current.parent
    return Path.cwd()


def check_progress():
    """Check current preprocessing progress."""
    root = get_project_root()
    registry_path = root / "data" / "processed" / "image" / "sample_registry.csv"
    
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}")
        return 1
    
    print("=" * 80)
    print("AEGIS IMAGE PREPROCESSING PROGRESS")
    print("=" * 80)
    
    # Read registry
    df = pd.read_csv(registry_path, low_memory=False)
    status_counts = df["status"].value_counts().to_dict()
    
    raw = status_counts.get("RAW", 0)
    processed = status_counts.get("PROCESSED", 0)
    failed = status_counts.get("FAILED", 0)
    processing = status_counts.get("PROCESSING", 0)
    total = len(df)
    
    completion_pct = (processed / total * 100) if total > 0 else 0
    
    print(f"\nTotal samples:    {total:,}")
    print(f"PROCESSED:        {processed:,} ({completion_pct:.2f}%)")
    print(f"RAW (pending):    {raw:,}")
    print(f"FAILED:           {failed}")
    print(f"PROCESSING:       {processing}")
    
    if raw > 0:
        print(f"\nRemaining:        {raw:,} samples")
        print(f"Estimated time:   ~{raw / 3.5 / 3600:.1f} hours at 3.5 img/s")
    else:
        print("\n✓ PREPROCESSING COMPLETE!")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(check_progress())
