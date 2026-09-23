"""Extract video frames from FaceForensics++_C23 and build video manifest.

Reads raw .mp4 files from FaceForensics++_C23/, extracts uniformly sampled
frames, saves to data/processed/video/frames/<video_id>/, and writes the
canonical video manifest at data/processed/video/manifest.csv.

Usage::

    python scripts/extract_video_frames.py

This must be run before generate_video_splits.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Could not locate AEGIS project root")


PROJECT_ROOT = find_project_root()

FF_ROOT = PROJECT_ROOT / "FaceForensics++_C23"
FRAMES_ROOT = PROJECT_ROOT / "data" / "processed" / "video" / "frames"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "video" / "manifest.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "video" / "manifest_summary.json"

FRAMES_PER_VIDEO = 10  # uniform temporal sampling

# Generator directory → (label, generator_id)
GENERATOR_MAP = {
    "original": ("real", "original"),
    "Deepfakes": ("fake", "Deepfakes"),
    "Face2Face": ("fake", "Face2Face"),
    "FaceShifter": ("fake", "FaceShifter"),  # used as unseen generator
    "FaceSwap": ("fake", "FaceSwap"),
    "NeuralTextures": ("fake", "NeuralTextures"),
}

MANIFEST_COLUMNS = (
    "video_id",
    "frame_path",
    "label",
    "identity_id",
    "generator",
    "original_source",
    "file_hash",
    "dataset",
    "split",
)


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
    """Extract n uniformly-sampled frames; returns list of saved paths."""
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.error("OpenCV not available. Install via: pip install opencv-python-headless")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing frames
    existing = sorted(out_dir.glob("*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(existing) >= n_frames:
        logger.debug("Frames already extracted for %s", video_id)
        return existing[:n_frames]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open: %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    indices = [min(idx, total_frames - 1) for idx in indices]

    saved: list[Path] = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_resized = cv2.resize(frame, (224, 224))
        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(out_path)

    cap.release()
    return saved


def main() -> int:
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    stats: dict[str, int] = {}
    total_videos = 0
    skipped = 0

    for gen_dir_name, (label, generator_id) in GENERATOR_MAP.items():
        gen_dir = FF_ROOT / gen_dir_name
        if not gen_dir.is_dir():
            logger.warning("Generator directory missing: %s (skipping)", gen_dir)
            continue

        videos = sorted(gen_dir.glob("*.mp4"))
        logger.info(
            "Processing %s: %d videos (label=%s, generator=%s)",
            gen_dir_name,
            len(videos),
            label,
            generator_id,
        )

        gen_rows = 0
        for video_path in videos:
            video_id = f"{generator_id}__{video_path.stem}"
            out_dir = FRAMES_ROOT / video_id

            frame_paths = extract_frames_opencv(
                video_path, out_dir, FRAMES_PER_VIDEO, video_id
            )
            if not frame_paths:
                skipped += 1
                continue

            total_videos += 1
            # Use first frame's hash as video representative hash
            file_hash = file_sha256(frame_paths[0])

            for frame_path in frame_paths:
                rel_path = frame_path.relative_to(PROJECT_ROOT).as_posix()
                manifest_rows.append({
                    "video_id": video_id,
                    "frame_path": rel_path,
                    "label": label,
                    "identity_id": video_path.stem,
                    "generator": generator_id,
                    "original_source": str(video_path.relative_to(PROJECT_ROOT).as_posix()),
                    "file_hash": file_hash,
                    "dataset": "FaceForensics++_C23",
                    "split": "unseen" if generator_id == "FaceShifter" else "seen",
                })
                gen_rows += 1

        stats[generator_id] = gen_rows
        logger.info("  → %d frame records", gen_rows)

    if not manifest_rows:
        logger.error("No frames extracted. Aborting.")
        return 1

    # Write manifest atomically
    temp_manifest = MANIFEST_PATH.with_suffix(".csv.tmp")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with temp_manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(manifest_rows)
    temp_manifest.replace(MANIFEST_PATH)

    logger.info("Manifest written: %s (%d rows)", MANIFEST_PATH, len(manifest_rows))

    # Write summary
    summary = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_videos": total_videos,
        "total_frame_records": len(manifest_rows),
        "skipped_videos": skipped,
        "frames_per_video": FRAMES_PER_VIDEO,
        "generator_frame_counts": stats,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("Summary: %s", SUMMARY_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
