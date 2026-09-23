"""Populate data/processed/image/splits/test_unseen.csv from FaceForensics++ C23.

Extracts uniformly-sampled frames from FaceForensics++_C23/ video directories
and writes rows conforming to the existing image split CSV schema.

Usage::

    python scripts/populate_image_test_unseen.py

Generators mapped:
    Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures → fake (unseen)
    original → real (unseen authentic)
"""

from __future__ import annotations

import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Could not locate AEGIS project root")


PROJECT_ROOT = find_project_root()

# Paths
FF_ROOT = PROJECT_ROOT / "FaceForensics++_C23"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "image" / "splits"
FRAMES_DIR = PROJECT_ROOT / "data" / "processed" / "image" / "unseen_frames"
TEST_UNSEEN_CSV = OUTPUT_DIR / "test_unseen.csv"

# Schema matches existing split CSVs
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

# Generator config: directory_name -> (label, generator_id, manipulation_method)
GENERATOR_MAP = {
    "Deepfakes": ("fake", "deepfakes", "deepfakes"),
    "Face2Face": ("fake", "face2face", "face2face"),
    "FaceShifter": ("fake", "faceshifter", "faceshifter"),
    "FaceSwap": ("fake", "faceswap", "faceswap"),
    "NeuralTextures": ("fake", "neuraltextures", "neuraltextures"),
    "original": ("real", "original_ff", "none"),
}

FRAMES_PER_VIDEO = 5  # uniform temporal sampling
CHUNK_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def extract_frames_opencv(
    video_path: Path,
    out_dir: Path,
    n_frames: int,
    video_id: str,
) -> list[Path]:
    """Extract n uniformly-sampled frames from a video using OpenCV."""
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.error("OpenCV not available. Install via: pip install opencv-python-headless")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        logger.warning("Zero frames in: %s", video_path)
        return []

    # Compute uniformly spaced frame indices
    indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    # Cap at total_frames - 1
    indices = [min(idx, total_frames - 1) for idx in indices]

    saved_paths: list[Path] = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Resize to 224×224 for consistency with image training data
        frame_resized = cv2.resize(frame, (224, 224))
        out_path = out_dir / f"{video_id}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


def iter_generator_records(
    generator_name: str,
    label: str,
    generator_id: str,
    manipulation_method: str,
) -> Iterator[dict[str, str]]:
    """Yield CSV row dicts for all videos in a FaceForensics++ generator directory."""
    gen_dir = FF_ROOT / generator_name
    if not gen_dir.is_dir():
        logger.warning("Directory not found, skipping: %s", gen_dir)
        return

    videos = sorted(gen_dir.glob("*.mp4"))
    logger.info("Processing %d videos from %s", len(videos), generator_name)

    for video_path in videos:
        video_id = video_path.stem  # e.g. "000_003"
        out_dir = FRAMES_DIR / generator_name / video_id

        # Check if frames already extracted
        existing = list(out_dir.glob("*.jpg")) if out_dir.exists() else []
        if len(existing) >= FRAMES_PER_VIDEO:
            frame_paths = sorted(existing)[:FRAMES_PER_VIDEO]
        else:
            frame_paths = extract_frames_opencv(video_path, out_dir, FRAMES_PER_VIDEO, video_id)
            if not frame_paths:
                continue

        for frame_path in frame_paths:
            rel_path = frame_path.relative_to(PROJECT_ROOT).as_posix()
            file_hash = file_sha256(frame_path)
            sample_id = f"ff_unseen:{generator_id}:{video_id}:{frame_path.stem}"

            yield {
                "sample_id": sample_id,
                "path": rel_path,
                "label": label,
                "identity_key": video_id,
                "generator": generator_id,
                "manipulation_method": manipulation_method,
                "original_source": str(video_path.relative_to(PROJECT_ROOT).as_posix()),
                "source_image_key": video_id,
                "file_hash": file_hash,
                "split_role": "test_unseen",
                "upstream_split": "test_unseen",
            }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []

    for generator_name, (label, generator_id, manipulation_method) in GENERATOR_MAP.items():
        rows = list(
            iter_generator_records(generator_name, label, generator_id, manipulation_method)
        )
        logger.info("  → %d frame records from %s", len(rows), generator_name)
        all_rows.extend(rows)

    if not all_rows:
        logger.error(
            "No frames were extracted. Ensure FaceForensics++_C23/ videos exist "
            "and opencv-python-headless is installed."
        )
        return 1

    # Write atomically
    temp_path = TEST_UNSEEN_CSV.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SPLIT_COLUMNS))
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda r: r["sample_id"]))
    temp_path.replace(TEST_UNSEEN_CSV)

    logger.info(
        "Wrote %d rows to %s",
        len(all_rows),
        TEST_UNSEEN_CSV.relative_to(PROJECT_ROOT),
    )

    # Print summary
    from collections import Counter
    label_counts = Counter(r["label"] for r in all_rows)
    gen_counts = Counter(r["generator"] for r in all_rows)
    logger.info("Label distribution: %s", dict(label_counts))
    logger.info("Generator distribution: %s", dict(gen_counts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
