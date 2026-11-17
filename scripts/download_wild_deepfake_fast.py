"""Fast WildDeepfake Downloader using Hugging Face CDN.

Downloads balanced video face-sequence archives for real and fake classes,
extracts them directly into data/raw/video/wild_deepfake/, and generates
a dataset manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WildDeepfake-Downloader")

BASE_URL = "https://huggingface.co/api/datasets/xingjunm/WildDeepfake/tree/main/deepfake_in_the_wild"
RESOLVE_URL = "https://huggingface.co/datasets/xingjunm/WildDeepfake/resolve/main/deepfake_in_the_wild"
DEFAULT_OUT_DIR = Path("data/raw/video/wild_deepfake")


def get_available_archives(split: str) -> List[str]:
    """Fetch list of available tar.gz files from HF API via curl."""
    api_url = f"{BASE_URL}/{split}"
    cmd = ["curl.exe", "-s", api_url]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    items = json.loads(res.stdout)
    archives = [
        item["path"].split("/")[-1]
        for item in items
        if item.get("type") == "file" and item["path"].endswith(".tar.gz")
    ]
    return sorted(archives)


def download_file(url: str, dest_path: Path) -> bool:
    """Download single archive via curl.exe."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl.exe", "-L", "-s", "-o", str(dest_path), url]
    res = subprocess.run(cmd)
    return res.returncode == 0 and dest_path.is_file() and dest_path.stat().st_size > 0


def extract_archive(tar_path: Path, dest_dir: Path) -> int:
    """Extract face sequences from tar archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.lower().endswith((".png", ".jpg", ".jpeg")):
                tar.extract(member, path=dest_dir, filter="data")
                count += 1
    return count


def run_download(num_archives: int = 15, split_type: str = "train", output_dir: Path = DEFAULT_OUT_DIR) -> dict:
    """Download and extract a balanced set of real and fake video archives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    summary = {"real_frames": 0, "fake_frames": 0, "total_frames": 0}
    splits = [("real", f"real_{split_type}"), ("fake", f"fake_{split_type}")]

    for label, split in splits:
        logger.info(f"Fetching archives list for {label} ({split})...")
        available = get_available_archives(split)
        selected = available[:num_archives]
        logger.info(f"Selected {len(selected)} archive(s) for {label}: {selected[:5]}...")

        target_dir = output_dir / label
        total_extracted = 0

        for idx, filename in enumerate(selected, 1):
            url = f"{RESOLVE_URL}/{split}/{filename}"
            local_tar = temp_dir / f"{label}_{filename}"
            logger.info(f"  [{label} {idx}/{len(selected)}] Downloading {filename}...")
            
            if download_file(url, local_tar):
                extracted = extract_archive(local_tar, target_dir)
                total_extracted += extracted
                logger.info(f"    Extracted {extracted} frames -> {target_dir}")
                local_tar.unlink(missing_ok=True)
            else:
                logger.error(f"    Failed to download {filename}")

        if label == "real":
            summary["real_frames"] = total_extracted
        else:
            summary["fake_frames"] = total_extracted
        summary["total_frames"] += total_extracted

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Download & extraction complete! Summary: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast WildDeepfake Downloader")
    parser.add_argument(
        "--archives-per-class",
        type=int,
        default=15,
        help="Number of archive parts to download per class (each is ~25-35MB)"
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split to download (train or test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory"
    )
    args = parser.parse_args()

    run_download(
        num_archives=args.archives_per_class,
        split_type=args.split_type,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
