"""
AEGIS Image Quality Validation
================================
Validates quality of all PROCESSED image artifacts.
Checks:
- file decodable
- dimensions >= 224x224
- crop is not corrupted/empty
- label/generator validity
- hash matching

Produces:
  reports/image_quality_report.json
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import sys

import pandas as pd
import numpy as np
import cv2

def find_project_root() -> Path:
    cwd = Path(__file__).resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise FileNotFoundError("Cannot locate AEGIS project root")

PROJECT_ROOT = find_project_root()
REGISTRY_PATH = PROJECT_ROOT / "data/processed/image/sample_registry.csv"
REPORT_PATH = PROJECT_ROOT / "reports/image_quality_report.json"

KNOWN_LABELS = {"real", "fake"}
KNOWN_GENERATORS = {"ffhq_authentic", "stylegan", "unknown"}
MIN_DIM = 224

def sha256_file(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def main(dry_run=False):
    print("=" * 70)
    print("AEGIS Image Quality Validation")
    print("=" * 70)
    
    registry_df = pd.read_csv(REGISTRY_PATH, low_memory=False)
    processed_df = registry_df[registry_df["status"] == "PROCESSED"]
    
    print(f"Validating {len(processed_df)} PROCESSED samples...")
    
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_processed": len(processed_df),
        "valid": 0,
        "invalid": 0,
        "errors": []
    }
    
    for _, row in processed_df.iterrows():
        sid = row["sample_id"]
        errors = []
        
        # 1. Paths exist
        crop_path = PROJECT_ROOT / row["crop_path"]
        norm_path = PROJECT_ROOT / row["processed_path"]
        
        if not crop_path.exists():
            errors.append(f"crop_missing:{row['crop_path']}")
        if not norm_path.exists():
            errors.append(f"npy_missing:{row['processed_path']}")
            
        if errors:
            results["invalid"] += 1
            results["errors"].append({"sample_id": sid, "errors": errors})
            continue
            
        # 2. Hash match
        expected_hash = str(row["file_hash"])
        actual_hash = sha256_file(crop_path)
        if expected_hash and actual_hash != expected_hash:
            errors.append(f"hash_mismatch:expected={expected_hash[:8]},actual={actual_hash[:8]}")
            
        # 3. Label/generator validity
        if str(row["label"]) not in KNOWN_LABELS:
            errors.append(f"invalid_label:{row['label']}")
        if str(row["generator"]) not in KNOWN_GENERATORS:
            errors.append(f"invalid_generator:{row['generator']}")
            
        # 4. Image decoding & dims
        img = cv2.imread(str(crop_path))
        if img is None:
            errors.append("crop_undecodable")
        else:
            h, w = img.shape[:2]
            if h < MIN_DIM or w < MIN_DIM:
                errors.append(f"dimensions_too_small:{w}x{h}")
                
        # 5. NPY checks
        try:
            arr = np.load(norm_path)
            if arr.shape != (3, MIN_DIM, MIN_DIM) and arr.shape != (MIN_DIM, MIN_DIM, 3):
                 errors.append(f"invalid_npy_shape:{arr.shape}")
            if np.isnan(arr).any():
                 errors.append("npy_contains_nan")
        except Exception as e:
            errors.append(f"npy_load_error:{e}")
            
        if errors:
            results["invalid"] += 1
            results["errors"].append({"sample_id": sid, "errors": errors})
        else:
            results["valid"] += 1

    print(f"Validation complete: {results['valid']} valid, {results['invalid']} invalid.")
    
    if not dry_run:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(results, indent=2))
        print(f"Written: {REPORT_PATH.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
