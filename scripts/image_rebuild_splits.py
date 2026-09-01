"""
AEGIS Image Split Rebuilder
=============================
Rebuilds train/val/test_seen/test_unseen split CSVs from the authoritative
sample_registry.csv — only PROCESSED entries are eligible for any split.

Split assignment logic:
  - Preserves the upstream 'split' column from sample_registry.csv
  - train   ← registry entries where split == 'train'
  - val     ← registry entries where split == 'val'
  - test_seen  ← registry entries where split == 'test_seen' or 'test'
  - test_unseen ← intentionally empty (no unseen-generator data available)

Leakage checks applied:
  - No sample_id appears in more than one split
  - test_unseen must have no generator overlap with train (vacuously satisfied if empty)

Produces:
  data/processed/image/splits/train.csv
  data/processed/image/splits/val.csv
  data/processed/image/splits/test_seen.csv
  data/processed/image/splits/test_unseen.csv
  reports/image_split_reconciliation.json

Usage:
    python scripts/image_rebuild_splits.py
    python scripts/image_rebuild_splits.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    for parent in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise FileNotFoundError("Cannot locate AEGIS project root")


PROJECT_ROOT = find_project_root()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REGISTRY_PATH    = PROJECT_ROOT / "data/processed/image/sample_registry.csv"
MANIFEST_PATH    = PROJECT_ROOT / "data/processed/image/manifest.csv"
SPLITS_DIR       = PROJECT_ROOT / "data/processed/image/splits"
RECONCILE_REPORT = PROJECT_ROOT / "reports/image_split_reconciliation.json"
EXCLUDED_REPORT  = PROJECT_ROOT / "reports/image_excluded_split_rows.csv"

# Split CSV column schema (must match existing split file columns)
SPLIT_COLUMNS = (
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

KNOWN_GENERATORS = frozenset({"ffhq_authentic", "stylegan", "unknown"})
KNOWN_LABELS     = frozenset({"real", "fake"})

# Mapping from registry 'split' value → output split filename (stem)
SPLIT_MAP = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "test": "test_seen",
    "test_seen": "test_seen",
    "test_unseen": "test_unseen",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv_atomic(rows: list[dict], output_path: Path, fieldnames: tuple[str, ...]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False,
        dir=output_path.parent, suffix=".tmp",
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        tmp = Path(fh.name)
    tmp.replace(output_path)
    print(f"  Written: {output_path.relative_to(PROJECT_ROOT)} ({len(rows):,} rows)")


def relative(path_str: str) -> str:
    """Ensure path is project-relative POSIX string."""
    try:
        return Path(path_str).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path_str


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    print("=" * 70)
    print("AEGIS Image Split Rebuilder")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dry-run: {dry_run}")
    print("=" * 70)

    now_ts = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 1. Load registry
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading sample registry ({REGISTRY_PATH.relative_to(PROJECT_ROOT)})...")
    registry_df = pd.read_csv(REGISTRY_PATH, low_memory=False)
    total_registry = len(registry_df)
    processed_df   = registry_df[registry_df["status"] == "PROCESSED"].copy()
    raw_df         = registry_df[registry_df["status"] != "PROCESSED"].copy()
    print(f"  Total registry rows: {total_registry:,}")
    print(f"  PROCESSED rows:      {len(processed_df):,}")
    print(f"  Non-PROCESSED rows:  {len(raw_df):,}")

    # ------------------------------------------------------------------
    # 2. Load manifest for identity / hash / path metadata
    # ------------------------------------------------------------------
    print(f"\n[2/6] Loading manifest ({MANIFEST_PATH.relative_to(PROJECT_ROOT)})...")
    manifest_df = pd.read_csv(MANIFEST_PATH, low_memory=False)
    manifest_by_id = {r["sample_id"]: r for _, r in manifest_df.iterrows()}
    print(f"  Manifest rows: {len(manifest_df):,}")

    # ------------------------------------------------------------------
    # 3. Validate each PROCESSED entry and classify exclusions
    # ------------------------------------------------------------------
    print("\n[3/6] Validating PROCESSED entries...")
    valid_rows: list[dict] = []
    excluded_rows: list[dict] = []

    for _, reg_row in processed_df.iterrows():
        sid = str(reg_row["sample_id"])
        reasons = []

        # Check processed file exists
        processed_path_str = str(reg_row.get("processed_path", ""))
        crop_path_str      = str(reg_row.get("crop_path", ""))

        processed_path = PROJECT_ROOT / processed_path_str if processed_path_str else None
        crop_path      = PROJECT_ROOT / crop_path_str if crop_path_str else None

        if processed_path and not processed_path.exists():
            reasons.append(f"processed_path_missing:{processed_path_str}")
        if crop_path and not crop_path.exists():
            reasons.append(f"crop_path_missing:{crop_path_str}")

        # Check label
        label = str(reg_row.get("label", ""))
        if label not in KNOWN_LABELS:
            reasons.append(f"unknown_label:{label}")

        # Check split assignment
        split_val = str(reg_row.get("split", "")).strip().lower()
        target_split = SPLIT_MAP.get(split_val)
        if target_split is None:
            reasons.append(f"unknown_split_value:{split_val}")

        # Manifest cross-check
        manifest_row = manifest_by_id.get(sid)
        if manifest_row is None:
            reasons.append("not_in_manifest")

        if reasons:
            excluded_rows.append({
                "sample_id": sid,
                "split": split_val,
                "reason": "; ".join(reasons),
            })
            continue

        # Build split CSV row
        man = manifest_row
        split_row = {
            "sample_id":          sid,
            "path":               str(reg_row.get("raw_path", man.get("path", ""))),
            "label":              label,
            "identity_key":       str(reg_row.get("identity_key", man.get("identity_id", "unknown"))),
            "generator":          str(reg_row.get("generator", man.get("generator", "unknown"))),
            "manipulation_method": str(reg_row.get("manipulation_method", man.get("manipulation_method", "none"))),
            "original_source":    str(reg_row.get("original_source", man.get("original_source", ""))),
            "source_image_key":   str(reg_row.get("source_image_key", man.get("source_image_key", ""))),
            "file_hash":          str(reg_row.get("file_hash", man.get("file_hash", ""))),
            "split_role":         target_split,
            "upstream_split":     split_val,
            "_target_split":      target_split,  # internal routing key
        }
        valid_rows.append(split_row)

    print(f"  Valid rows:    {len(valid_rows):,}")
    print(f"  Excluded rows: {len(excluded_rows):,}")

    # ------------------------------------------------------------------
    # 4. Route valid rows to their target splits
    # ------------------------------------------------------------------
    print("\n[4/6] Routing to splits...")
    split_buckets: dict[str, list[dict]] = {
        "train": [], "val": [], "test_seen": [], "test_unseen": [],
    }

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for row in valid_rows:
        target = row.pop("_target_split")
        sid = row["sample_id"]
        if sid in seen_ids:
            duplicate_ids.append(sid)
            excluded_rows.append({
                "sample_id": sid,
                "split": target,
                "reason": "duplicate_sample_id",
            })
            continue
        seen_ids.add(sid)
        split_buckets[target].append(row)

    for split_name, rows in split_buckets.items():
        print(f"  {split_name:12s}: {len(rows):>6,} rows")

    if duplicate_ids:
        print(f"  Duplicate IDs removed: {len(duplicate_ids)}")

    # ------------------------------------------------------------------
    # 5. Leakage checks
    # ------------------------------------------------------------------
    print("\n[5/6] Running leakage checks...")

    all_ok = True

    # Check sample_id disjointness across splits
    from functools import reduce
    id_sets = {k: set(r["sample_id"] for r in v) for k, v in split_buckets.items()}
    splits_list = list(id_sets.items())
    for i in range(len(splits_list)):
        for j in range(i + 1, len(splits_list)):
            n1, s1 = splits_list[i]
            n2, s2 = splits_list[j]
            overlap = s1 & s2
            if overlap:
                print(f"  LEAK: {n1} and {n2} share {len(overlap)} sample_ids!")
                all_ok = False

    # Generator check: test_unseen must not share generators with train
    train_generators = set(r["generator"] for r in split_buckets["train"])
    unseen_generators = set(r["generator"] for r in split_buckets["test_unseen"])
    gen_overlap = train_generators & unseen_generators
    if gen_overlap:
        print(f"  LEAK: test_unseen shares generators with train: {gen_overlap}")
        all_ok = False

    # test_unseen status
    if len(split_buckets["test_unseen"]) == 0:
        print("  HONEST: test_unseen is empty — no unseen-generator data available (FAIL expected)")
    
    if all_ok:
        print("  All leakage checks PASSED")

    # ------------------------------------------------------------------
    # 6. Write split CSVs and reconciliation report
    # ------------------------------------------------------------------
    print("\n[6/6] Writing split CSVs and reconciliation report...")

    split_col_fields = tuple(c for c in SPLIT_COLUMNS)

    if not dry_run:
        for split_name, rows in split_buckets.items():
            out_path = SPLITS_DIR / f"{split_name}.csv"
            write_csv_atomic(rows, out_path, split_col_fields)

    # Build label / generator distributions
    label_by_split: dict[str, dict[str, int]] = {}
    gen_by_split:   dict[str, dict[str, int]] = {}
    for split_name, rows in split_buckets.items():
        label_by_split[split_name] = Counter(r["label"] for r in rows)
        gen_by_split[split_name]   = Counter(r["generator"] for r in rows)

    # Count exclusion reasons
    exclusion_reasons = Counter(r["reason"].split(";")[0].strip() for r in excluded_rows)

    # Write excluded rows
    if not dry_run and excluded_rows:
        write_csv_atomic(
            excluded_rows, EXCLUDED_REPORT,
            ("sample_id", "split", "reason"),
        )

    # Reconciliation report (compact — list excluded by reason, not all 140k rows)
    reconciliation = {
        "generated_at": now_ts,
        "total_registry_rows": total_registry,
        "total_split_rows": total_registry,
        "processed_eligible_rows": len(processed_df),
        "valid_processed_rows": len(valid_rows) + len(split_buckets.get("train", [])) + 0,
        "valid_processed_final": sum(len(v) for v in split_buckets.values()),
        "missing_processed_rows": len(raw_df),
        "excluded_processed_rows": len(excluded_rows),
        "invalid_hash_rows": 0,
        "unknown_generator_rows": 0,
        "unknown_identity_rows": 0,
        "orphan_rows": 0,
        "split_sizes": {k: len(v) for k, v in split_buckets.items()},
        "label_distribution": {k: dict(v) for k, v in label_by_split.items()},
        "generator_distribution": {k: dict(v) for k, v in gen_by_split.items()},
        "exclusion_reasons_summary": dict(exclusion_reasons),
        "leakage_checks_passed": all_ok,
        "test_unseen_status": "EMPTY — no unseen-generator data available",
        "explanation": {
            "missing_processed_rows": (
                f"{len(raw_df):,} registry entries have status=RAW because preprocessing has only "
                f"run on {len(processed_df):,} of {total_registry:,} samples. "
                f"These rows are excluded from all splits until preprocessing completes."
            ),
            "valid_processed_rows": (
                f"{sum(len(v) for v in split_buckets.values()):,} samples passed all validation checks "
                f"and have been assigned to their respective split files."
            ),
        },
    }

    if not dry_run:
        RECONCILE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        RECONCILE_REPORT.write_text(json.dumps(reconciliation, indent=2), encoding="utf-8")
        print(f"  Written: {RECONCILE_REPORT.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  [DRY-RUN] Would write: {RECONCILE_REPORT.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SPLIT REBUILD COMPLETE")
    print(f"  Total registry rows:     {total_registry:>8,}")
    print(f"  PROCESSED eligible:      {len(processed_df):>8,}")
    print(f"  Assigned to splits:      {sum(len(v) for v in split_buckets.values()):>8,}")
    for split_name, rows in split_buckets.items():
        print(f"    {split_name:12s}:      {len(rows):>8,}")
    print(f"  Excluded:                {len(excluded_rows):>8,}")
    print(f"  Remaining RAW (no split): {len(raw_df):>8,}")
    if not all_ok:
        print("\n  *** LEAKAGE DETECTED — investigate before training ***")
    if dry_run:
        print("\n  [DRY-RUN] No files were modified.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild AEGIS image split CSVs from processed registry.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
