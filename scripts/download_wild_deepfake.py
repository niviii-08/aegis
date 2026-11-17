"""WildDeepfake Dataset Downloader and Extractor.

Supports downloading curated samples or full splits from Hugging Face
into data/raw/image/wild_deepfake.
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "xingjunm/WildDeepfake"
DEFAULT_OUT_DIR = Path("data/raw/image/wild_deepfake")


def get_archive_list(api: HfApi, split: str, limit: int = 0) -> List[str]:
    """Retrieve list of tar.gz archives for a split."""
    info = api.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    prefix = f"deepfake_in_the_wild/{split}/"
    files = [
        s.rfilename for s in info.siblings
        if s.rfilename.startswith(prefix) and s.rfilename.endswith(".tar.gz")
    ]
    if limit > 0:
        return files[:limit]
    return files


def extract_tar_archive(tar_path: Path, dest_dir: Path) -> int:
    """Extract tar.gz archive safely into destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.lower().endswith((".png", ".jpg", ".jpeg")):
                tar.extract(member, path=dest_dir, filter="data")
                count += 1
    return count


def download_split_archives(
    split: str,
    target_label: str,
    limit: int,
    output_dir: Path
) -> int:
    """Download and extract archives for a given split."""
    api = HfApi()
    files = get_archive_list(api, split, limit=limit)
    dest_dir = output_dir / target_label
    total_extracted = 0

    print(f"[{split}] Downloading {len(files)} archive(s)...")
    for idx, rfilename in enumerate(files, start=1):
        print(f"  ({idx}/{len(files)}) Fetching {rfilename}...")
        local_tar = hf_hub_download(
            repo_id=REPO_ID,
            filename=rfilename,
            repo_type="dataset"
        )
        extracted = extract_tar_archive(Path(local_tar), dest_dir)
        total_extracted += extracted
        print(f"    Extracted {extracted} images -> {dest_dir}")

    return total_extracted


def main() -> None:
    """CLI entrypoint for WildDeepfake download."""
    parser = argparse.ArgumentParser(description="WildDeepfake Downloader")
    parser.add_argument(
        "--mode",
        choices=["sample", "test", "full"],
        default="sample",
        help="sample: 5 archives per class (~600MB); test: full test split (~9.5GB); full: entire ~68GB"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of archives per class for sample mode (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Destination directory"
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    limit = args.limit if args.mode == "sample" else 0

    if args.mode in ("sample", "test"):
        real_extracted = download_split_archives("real_test", "real", limit, out_path)
        fake_extracted = download_split_archives("fake_test", "fake", limit, out_path)
    else:
        real_extracted = download_split_archives("real_train", "real", limit, out_path)
        fake_extracted = download_split_archives("fake_train", "fake", limit, out_path)

    print("\nDownload & Extraction Complete!")
    print(f"Real images: {real_extracted}")
    print(f"Fake images: {fake_extracted}")
    print(f"Output directory: {out_path.resolve()}")


if __name__ == "__main__":
    main()
