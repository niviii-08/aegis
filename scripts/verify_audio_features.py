"""Verify audio mel-spectrogram features are present.

Usage::
    python scripts/verify_audio_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Could not locate AEGIS project root")

PROJECT_ROOT = find_project_root()

def main() -> int:
    metadata_path = PROJECT_ROOT / "reports" / "audio" / "preprocessing_metadata_mel_spectrogram.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata file missing: {metadata_path}")
        return 1
        
    total_found = 0
    missing = 0
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success":
                path_str = row.get("processed_feature_path")
                if path_str:
                    full_path = PROJECT_ROOT / path_str
                    if full_path.exists():
                        total_found += 1
                    else:
                        missing += 1
                        
    logger.info(f"Mel-spectrogram check: {total_found} features found, {missing} missing.")
    if total_found == 0:
        logger.error("No features found! Need to run preprocessing.")
        return 1
        
    logger.info("Audio features verified successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
