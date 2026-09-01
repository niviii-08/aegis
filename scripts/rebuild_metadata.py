"""Rebuild preprocessing metadata from actual disk state.

This script reconciles metadata.csv with the actual crop/normalized files on disk.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "data" / "processed" / "image" / "preprocessing"
CROPS_DIR = PREPROCESSING_DIR / "crops"
NORMALIZED_DIR = PREPROCESSING_DIR / "normalized"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "image" / "manifest.csv"
METADATA_PATH = PREPROCESSING_DIR / "metadata.csv"
FAILURES_PATH = PROJECT_ROOT / "reports" / "preprocessing_failures.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "preprocessing_summary.json"

METADATA_COLUMNS = [
    "sample_id",
    "original_path",
    "processed_crop_path",
    "processed_normalized_path",
    "preprocessing_version",
    "detector",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "detection_confidence",
    "alignment_succeeded",
    "face_count",
    "face_width_pixels",
    "face_height_pixels",
    "source_width",
    "source_height",
    "status",
    "error_message",
    "processing_time_ms",
    "processed_at",
]

FAILURE_COLUMNS = [
    "sample_id",
    "original_path",
    "status",
    "error_message",
    "face_count",
    "detector",
    "preprocessing_version",
    "processed_at",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def try_read_normalized_meta(norm_path: Path) -> dict[str, Any]:
    """Try to extract metadata from a normalized .npy file."""
    try:
        data = np.load(norm_path, allow_pickle=False)
        # normalized is CHW float32; infer original size if possible
        # The metadata isn't stored in the npy, so we return empty
        return {}
    except Exception:
        return {}


def try_read_crop_meta(crop_path: Path) -> dict[str, Any]:
    """Try to extract metadata from a crop image."""
    try:
        with Image.open(crop_path) as img:
            w, h = img.size
        return {"crop_width": w, "crop_height": h}
    except Exception:
        return {}


def rebuild_metadata() -> None:
    logger.info("Rebuilding metadata.csv from actual disk state...")

    manifest = pd.read_csv(MANIFEST_PATH)
    existing_meta = pd.read_csv(METADATA_PATH) if METADATA_PATH.exists() else pd.DataFrame()
    existing_meta_by_id = {}
    if not existing_meta.empty:
        for _, row in existing_meta.iterrows():
            sid = row.get("sample_id", "")
            if sid:
                existing_meta_by_id[sid] = row.to_dict()

    # Index crops and normalized files
    crop_files = {c.stem: c for c in CROPS_DIR.glob("*.jpg")} if CROPS_DIR.exists() else {}
    norm_files = {n.stem: n for n in NORMALIZED_DIR.glob("*.npy")} if NORMALIZED_DIR.exists() else {}

    rows = []
    failure_rows = []
    stats = {
        "total": len(manifest),
        "already_in_metadata": 0,
        "recovered_from_disk": 0,
        "still_missing": 0,
        "failed": 0,
    }

    for _, mrow in manifest.iterrows():
        sample_id = mrow["sample_id"]
        stem = sample_id.replace(":", "__")
        original_path = mrow["path"]

        existing = existing_meta_by_id.get(sample_id)
        if existing is not None and existing.get("status") == "success":
            stats["already_in_metadata"] += 1
            rows.append({col: existing.get(col, "") for col in METADATA_COLUMNS})
            continue

        crop_path = crop_files.get(stem)
        norm_path = norm_files.get(stem)

        if crop_path and norm_path:
            # Recover from disk
            crop_meta = try_read_crop_meta(crop_path)
            existing = existing_meta_by_id.get(sample_id) or {}
            row = {
                "sample_id": sample_id,
                "original_path": original_path,
                "processed_crop_path": str(crop_path.resolve()),
                "processed_normalized_path": str(norm_path.resolve()),
                "preprocessing_version": existing.get("preprocessing_version", "unknown"),
                "detector": existing.get("detector", "mtcnn"),
                "bbox_x": existing.get("bbox_x", ""),
                "bbox_y": existing.get("bbox_y", ""),
                "bbox_w": existing.get("bbox_w", ""),
                "bbox_h": existing.get("bbox_h", ""),
                "detection_confidence": existing.get("detection_confidence", ""),
                "alignment_succeeded": existing.get("alignment_succeeded", "true"),
                "face_count": existing.get("face_count", "1"),
                "face_width_pixels": existing.get("face_width_pixels", crop_meta.get("crop_width", "")),
                "face_height_pixels": existing.get("face_height_pixels", crop_meta.get("crop_height", "")),
                "source_width": existing.get("source_width", ""),
                "source_height": existing.get("source_height", ""),
                "status": "success",
                "error_message": existing.get("error_message", ""),
                "processing_time_ms": existing.get("processing_time_ms", ""),
                "processed_at": existing.get("processed_at", datetime.now(timezone.utc).isoformat()),
            }
            rows.append(row)
            stats["recovered_from_disk"] += 1
        else:
            existing = existing_meta_by_id.get(sample_id) or {}
            row = {
                "sample_id": sample_id,
                "original_path": original_path,
                "processed_crop_path": str(crop_path.resolve()) if crop_path else "",
                "processed_normalized_path": str(norm_path.resolve()) if norm_path else "",
                "preprocessing_version": existing.get("preprocessing_version", "unknown"),
                "detector": existing.get("detector", "mtcnn"),
                "bbox_x": existing.get("bbox_x", ""),
                "bbox_y": existing.get("bbox_y", ""),
                "bbox_w": existing.get("bbox_w", ""),
                "bbox_h": existing.get("bbox_h", ""),
                "detection_confidence": existing.get("detection_confidence", ""),
                "alignment_succeeded": existing.get("alignment_succeeded", "false"),
                "face_count": existing.get("face_count", "0"),
                "face_width_pixels": existing.get("face_width_pixels", ""),
                "face_height_pixels": existing.get("face_height_pixels", ""),
                "source_width": existing.get("source_width", ""),
                "source_height": existing.get("source_height", ""),
                "status": "no_crop",
                "error_message": "no_crop_or_normalized_file_found",
                "processing_time_ms": existing.get("processing_time_ms", ""),
                "processed_at": existing.get("processed_at", datetime.now(timezone.utc).isoformat()),
            }
            rows.append(row)
            stats["still_missing"] += 1

            # Add to failures
            failure_rows.append({col: row.get(col, "") for col in FAILURE_COLUMNS})

    # Write metadata
    with METADATA_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Write failures
    with FAILURES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        writer.writerows(failure_rows)

    # Write summary
    summary = {
        "preprocessing_version": "unknown",
        "config_path": "configs/image_preprocessing.yaml",
        "manifest_path": str(MANIFEST_PATH),
        "detector": "mtcnn",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_images": stats["total"],
        "skipped_resumed": 0,
        "processed_this_run": stats["recovered_from_disk"],
        "successful_crops": stats["recovered_from_disk"],
        "failed_crops": stats["still_missing"],
        "no_face_cases": 0,
        "multi_face_cases": 0,
        "average_face_size_pixels": 0.0,
        "total_processing_time_seconds": 0.0,
        "notes": [
            f"Metadata rebuilt from disk state. {stats['recovered_from_disk']} samples recovered from existing crops.",
            f"{stats['still_missing']} samples have no processed artifacts.",
            "This is a metadata rebuild, not a full preprocessing run.",
        ],
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Rebuild complete: %s", stats)


if __name__ == "__main__":
    rebuild_metadata()
