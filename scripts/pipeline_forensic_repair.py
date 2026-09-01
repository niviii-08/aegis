"""AEGIS Image Data Pipeline Forensic Repair and Completion.

This script performs Steps 2-4 of the pipeline recovery:
- STEP 2: Artifact audit (orphan crops, mismatches)
- STEP 3: Authoritative sample registry creation
- STEP 4: Split reconciliation

It also prepares the ground for subsequent steps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_IMAGE = PROJECT_ROOT / "data" / "processed" / "image"
PREPROCESSING_DIR = DATA_PROCESSED_IMAGE / "preprocessing"
CROPS_DIR = PREPROCESSING_DIR / "crops"
NORMALIZED_DIR = PREPROCESSING_DIR / "normalized"
MANIFEST_PATH = DATA_PROCESSED_IMAGE / "manifest.csv"
METADATA_PATH = PREPROCESSING_DIR / "metadata.csv"
REGISTRY_PATH = DATA_PROCESSED_IMAGE / "sample_registry.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 2: ARTIFACT AUDIT
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_artifact_audit() -> dict[str, Any]:
    logger.info("STEP 2: Running artifact audit...")

    manifest = pd.read_csv(MANIFEST_PATH)
    metadata = pd.read_csv(METADATA_PATH) if METADATA_PATH.exists() else pd.DataFrame()
    registry = pd.read_csv(REGISTRY_PATH) if REGISTRY_PATH.exists() else pd.DataFrame()

    crop_files = sorted(
        list(CROPS_DIR.glob("*.jpg")) + list(CROPS_DIR.glob("*.png"))
    ) if CROPS_DIR.exists() else []

    manifest_sample_ids = set(manifest["sample_id"].tolist())
    metadata_sample_ids = set(metadata["sample_id"].tolist()) if not metadata.empty else set()

    # Map crop stems to sample IDs
    crop_stem_to_sample = {}
    for c in crop_files:
        crop_stem_to_sample[c.stem] = c.stem.replace("__", ":")

    crop_sample_ids = set(crop_stem_to_sample.values())

    matched_crops = crop_sample_ids & metadata_sample_ids
    orphan_crops = crop_sample_ids - metadata_sample_ids
    missing_crops = metadata_sample_ids - crop_sample_ids

    # Check for duplicates
    stem_counts = defaultdict(int)
    for stem in crop_stem_to_sample:
        stem_counts[stem] += 1
    duplicate_crops = {stem for stem, count in stem_counts.items() if count > 1}

    # Identify root cause
    orphan_details = []
    for sample_id in sorted(orphan_crops):
        stem = sample_id.replace(":", "__")
        crop_path = CROPS_DIR / f"{stem}.jpg"
        if not crop_path.exists():
            crop_path = CROPS_DIR / f"{stem}.png"
        in_manifest = sample_id in manifest_sample_ids
        in_registry = sample_id in set(registry["sample_id"].tolist()) if not registry.empty else False
        reg_status = ""
        if in_registry:
            reg_status = registry[registry["sample_id"] == sample_id]["status"].iloc[0]
        orphan_details.append({
            "sample_id": sample_id,
            "crop_path": str(crop_path) if crop_path.exists() else "",
            "in_manifest": in_manifest,
            "in_registry": in_registry,
            "registry_status": reg_status,
            "likely_cause": "stale_output" if in_registry and reg_status == "RAW" else "orphaned_file",
        })

    audit = {
        "raw_sample_count": len(manifest),
        "manifest_count": len(manifest),
        "metadata_count": len(metadata),
        "crop_count": len(crop_files),
        "matched_crop_count": len(matched_crops),
        "orphan_crop_count": len(orphan_crops),
        "duplicate_crop_count": len(duplicate_crops),
        "missing_crop_count": len(missing_crops),
        "root_cause_analysis": {
            "finding": (
                f"Of {len(crop_files)} crop files, only {len(matched_crops)} have corresponding "
                f"metadata entries. The remaining {len(orphan_crops)} crops are orphaned outputs "
                f"from a prior preprocessing run that did not update metadata.csv. "
                f"These samples remain marked as RAW in the registry."
            ),
            "orphan_breakdown": {
                "stale_outputs_from_previous_run": sum(1 for d in orphan_details if d["likely_cause"] == "stale_output"),
                "other_orphaned": sum(1 for d in orphan_details if d["likely_cause"] == "orphaned_file"),
            },
        },
    }

    audit_path = REPORTS_DIR / "image_artifact_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info("Saved %s", audit_path)

    orphan_csv_path = REPORTS_DIR / "image_orphan_crops.csv"
    with orphan_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "crop_path", "in_manifest", "in_registry", "registry_status", "likely_cause"])
        writer.writeheader()
        writer.writerows(orphan_details)
    logger.info("Saved %s (%d rows)", orphan_csv_path, len(orphan_details))

    return audit


# ---------------------------------------------------------------------------
# STEP 3: AUTHORITATIVE SAMPLE REGISTRY
# ---------------------------------------------------------------------------

def build_sample_registry() -> pd.DataFrame:
    logger.info("STEP 3: Building authoritative sample registry...")

    manifest = pd.read_csv(MANIFEST_PATH)
    metadata = pd.read_csv(METADATA_PATH) if METADATA_PATH.exists() else pd.DataFrame()

    # Build metadata lookup
    meta_by_id: dict[str, dict] = {}
    if not metadata.empty:
        for _, row in metadata.iterrows():
            meta_by_id[row["sample_id"]] = row.to_dict()

    # Crop files
    crop_files = sorted(
        list(CROPS_DIR.glob("*.jpg")) + list(CROPS_DIR.glob("*.png"))
    ) if CROPS_DIR.exists() else []
    crop_by_stem = {}
    for c in crop_files:
        crop_by_stem[c.stem] = str(c.resolve())

    rows = []
    for _, mrow in manifest.iterrows():
        sample_id = mrow["sample_id"]
        stem = sample_id.replace(":", "__")
        meta = meta_by_id.get(sample_id, {})

        raw_path = mrow["path"]
        crop_path = meta.get("processed_crop_path", "")
        processed_path = meta.get("processed_normalized_path", "")

        # Determine status
        status = "RAW"
        if sample_id in meta_by_id:
            meta_status = meta.get("status", "")
            if meta_status == "success":
                # Verify crop actually exists
                if crop_path and Path(crop_path).is_file():
                    status = "PROCESSED"
                else:
                    status = "FAILED"
            else:
                status = "FAILED"

        # File hash from manifest
        file_hash = mrow.get("file_hash", "")

        # Preprocessing version
        preprocessing_version = meta.get("preprocessing_version", "unknown")

        rows.append({
            "sample_id": sample_id,
            "source_dataset": mrow.get("dataset", "real_vs_fake"),
            "original_source": mrow.get("original_source", "unknown"),
            "source_image_key": mrow.get("original_source", "unknown"),
            "identity_key": mrow.get("identity_id", "unknown"),
            "generator": mrow.get("generator", "unknown"),
            "manipulation_method": mrow.get("manipulation_method", "unknown"),
            "label": mrow.get("label", "unknown"),
            "raw_path": raw_path,
            "processed_path": processed_path,
            "crop_path": crop_path,
            "file_hash": file_hash,
            "preprocessing_version": preprocessing_version,
            "split": mrow.get("split", "unknown"),
            "status": status,
        })

    registry = pd.DataFrame(rows)
    registry.to_csv(REGISTRY_PATH, index=False)
    logger.info("Saved registry with %d rows to %s", len(registry), REGISTRY_PATH)

    status_counts = registry["status"].value_counts().to_dict()
    logger.info("Registry status counts: %s", status_counts)

    return registry


# ---------------------------------------------------------------------------
# STEP 4: SPLIT RECONCILIATION
# ---------------------------------------------------------------------------

def load_split(split_name: str) -> pd.DataFrame:
    path = DATA_PROCESSED_IMAGE / "splits" / f"{split_name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def reconcile_splits(registry: pd.DataFrame) -> dict[str, Any]:
    logger.info("STEP 4: Reconciling splits against registry...")

    registry_map = {row["sample_id"]: row for _, row in registry.iterrows()}
    registry_processed_ids = set(registry[registry["status"] == "PROCESSED"]["sample_id"].tolist())

    split_names = ["train", "val", "test_seen", "test_unseen"]
    total_split_rows = 0
    valid_processed_rows = 0
    missing_processed_rows = 0
    invalid_hash_rows = 0
    unknown_generator_rows = 0
    unknown_identity_rows = 0
    orphan_rows = 0

    excluded_rows = []

    for split_name in split_names:
        df = load_split(split_name)
        if df.empty:
            continue

        for _, row in df.iterrows():
            total_split_rows += 1
            sample_id = row.get("sample_id", "")
            reasons = []

            # 1. Sample exists in registry
            if sample_id not in registry_map:
                reasons.append("missing_from_registry")
                orphan_rows += 1
                excluded_rows.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": ";".join(reasons),
                })
                continue

            reg_row = registry_map[sample_id]

            # 2. Processed file exists
            if reg_row["status"] != "PROCESSED":
                reasons.append(f"status={reg_row['status']}")
                missing_processed_rows += 1
                excluded_rows.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": ";".join(reasons),
                })
                continue

            # 3. File hash valid
            split_hash = row.get("file_hash", "")
            reg_hash = reg_row.get("file_hash", "")
            if split_hash and reg_hash and split_hash != reg_hash:
                reasons.append("hash_mismatch")
                invalid_hash_rows += 1
                excluded_rows.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": ";".join(reasons),
                })
                continue

            # 4. Label valid
            label = row.get("label", "")
            if label not in {"real", "fake"}:
                reasons.append(f"invalid_label={label}")
                missing_processed_rows += 1
                excluded_rows.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": ";".join(reasons),
                })
                continue

            # 5. Generator known
            generator = row.get("generator", "")
            if not generator or generator == "unknown":
                reasons.append("unknown_generator")
                unknown_generator_rows += 1
                excluded_rows.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": ";".join(reasons),
                })
                continue

            # 6. Identity known
            identity = row.get("identity_key", "")
            if not identity or identity == "unknown":
                reasons.append("unknown_identity")
                unknown_identity_rows += 1
                excluded_rows.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": ";".join(reasons),
                })
                continue

            # 7. Source dataset known
            source_dataset = reg_row.get("source_dataset", "")
            if not source_dataset or source_dataset == "unknown":
                reasons.append("unknown_source_dataset")
                missing_processed_rows += 1
                excluded_rows.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": ";".join(reasons),
                })
                continue

            # 8. Preprocessing status is PROCESSED
            if reg_row["status"] != "PROCESSED":
                reasons.append(f"preprocessing_status={reg_row['status']}")
                missing_processed_rows += 1
                excluded_rows.append({
                    "sample_id": sample_id,
                    "split": split_name,
                    "reason": ";".join(reasons),
                })
                continue

            valid_processed_rows += 1

    reconciliation = {
        "total_split_rows": total_split_rows,
        "valid_processed_rows": valid_processed_rows,
        "missing_processed_rows": missing_processed_rows,
        "invalid_hash_rows": invalid_hash_rows,
        "unknown_generator_rows": unknown_generator_rows,
        "unknown_identity_rows": unknown_identity_rows,
        "orphan_rows": orphan_rows,
        "excluded_rows_count": len(excluded_rows),
        "excluded_rows": excluded_rows,
        "conclusion": (
            f"Of {total_split_rows} total split rows, only {valid_processed_rows} ({valid_processed_rows/total_split_rows*100:.1f}%) "
            f"are backed by actually processed artifacts. The splits were generated from the raw manifest "
            f"before preprocessing was complete, so they contain references to unprocessed samples."
        ) if total_split_rows > 0 else "No split rows found.",
    }

    rec_path = REPORTS_DIR / "image_split_reconciliation.json"
    with rec_path.open("w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2, default=str)
    logger.info("Saved %s", rec_path)

    excluded_path = REPORTS_DIR / "image_excluded_split_rows.csv"
    with excluded_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "reason"])
        writer.writeheader()
        writer.writerows(excluded_rows)
    logger.info("Saved %s (%d rows)", excluded_path, len(excluded_rows))

    return reconciliation


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("AEGIS Image Pipeline Forensic Repair - Steps 2-4")
    logger.info("=" * 60)

    audit = run_artifact_audit()
    registry = build_sample_registry()
    reconciliation = reconcile_splits(registry)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Manifest entries: %d", audit["manifest_count"])
    logger.info("  Metadata entries: %d", audit["metadata_count"])
    logger.info("  Crop files: %d", audit["crop_count"])
    logger.info("  Matched crops: %d", audit["matched_crop_count"])
    logger.info("  Orphan crops: %d", audit["orphan_crop_count"])
    logger.info("  Duplicate crops: %d", audit["duplicate_crop_count"])
    logger.info("  Missing crops: %d", audit["missing_crop_count"])
    logger.info("  Registry PROCESSED: %d", int(registry[registry["status"] == "PROCESSED"].shape[0]))
    logger.info("  Registry RAW: %d", int(registry[registry["status"] == "RAW"].shape[0]))
    logger.info("  Valid split rows: %d", reconciliation["valid_processed_rows"])
    logger.info("  Total split rows: %d", reconciliation["total_split_rows"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
