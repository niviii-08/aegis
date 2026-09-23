"""Generate video experiment splits from the video manifest.

Reads data/processed/video/manifest.csv and configs/video_split.yaml,
assigns videos to train/val/test_seen/test_unseen splits, and writes split
CSVs to data/processed/video/splits/.

FaceShifter is the unseen generator (replaces celeb_df in config).

Usage::

    python scripts/generate_video_splits.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Could not locate AEGIS project root")


PROJECT_ROOT = find_project_root()

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "video" / "manifest.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "video" / "splits"
STATS_PATH = PROJECT_ROOT / "reports" / "video" / "split_statistics.json"

SPLIT_COLUMNS = (
    "video_id",
    "frame_path",
    "label",
    "identity_id",
    "generator",
    "original_source",
    "file_hash",
    "split_role",
)

# Generators assigned to each split
UNSEEN_GENERATORS = {"FaceShifter"}
SEEN_GENERATORS = {"original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}

# Target split ratios for seen videos (by video_id)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test_seen gets the remainder (~0.15)

SEED = 42


def load_manifest() -> list[dict[str, str]]:
    """Load video manifest rows."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Video manifest not found: {MANIFEST_PATH}\n"
            "Run scripts/extract_video_frames.py first."
        )
    rows = []
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def assign_splits(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Assign split_role to each row.

    Strategy:
    - FaceShifter → test_unseen
    - Seen generators → group by video_id, hash-assign to train/val/test_seen
    """
    # Group by video_id
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)

    # Separate unseen vs seen video IDs
    rng = random.Random(SEED)
    seen_video_ids = []
    unseen_video_ids = []

    for video_id, video_rows in by_video.items():
        gen = video_rows[0]["generator"]
        if gen in UNSEEN_GENERATORS:
            unseen_video_ids.append(video_id)
        else:
            seen_video_ids.append(video_id)

    # Shuffle seen videos deterministically
    seen_video_ids.sort()
    rng.shuffle(seen_video_ids)

    n = len(seen_video_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = set(seen_video_ids[:n_train])
    val_ids = set(seen_video_ids[n_train : n_train + n_val])
    test_seen_ids = set(seen_video_ids[n_train + n_val :])

    logger.info(
        "Split assignment: train=%d, val=%d, test_seen=%d videos | test_unseen=%d videos",
        len(train_ids),
        len(val_ids),
        len(test_seen_ids),
        len(unseen_video_ids),
    )

    assigned_rows = []
    for video_id, video_rows in by_video.items():
        if video_id in train_ids:
            role = "train"
        elif video_id in val_ids:
            role = "val"
        elif video_id in test_seen_ids:
            role = "test_seen"
        else:
            role = "test_unseen"

        for row in video_rows:
            new_row = dict(row)
            new_row["split_role"] = role
            assigned_rows.append(new_row)

    return assigned_rows


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SPLIT_COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["video_id"]):
            writer.writerow({col: row.get(col, "") for col in SPLIT_COLUMNS})
    temp.replace(path)


def main() -> int:
    rows = load_manifest()
    logger.info("Loaded %d manifest rows", len(rows))

    assigned = assign_splits(rows)

    # Group by split_role
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in assigned:
        by_split[row["split_role"]].append(row)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    label_counts = {}
    gen_counts = {}

    for role in ("train", "val", "test_seen", "test_unseen"):
        split_rows = by_split.get(role, [])
        out_path = SPLITS_DIR / f"{role}.csv"
        write_split_csv(out_path, split_rows)
        n = len(split_rows)
        split_counts[role] = n
        label_counts[role] = dict(Counter(r["label"] for r in split_rows))
        gen_counts[role] = dict(Counter(r["generator"] for r in split_rows))
        logger.info(
            "Wrote %s: %d rows | labels=%s",
            out_path.name,
            n,
            label_counts[role],
        )

    # Write statistics
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_manifest_rows": len(rows),
        "split_counts": split_counts,
        "label_counts_by_split": label_counts,
        "generator_counts_by_split": gen_counts,
        "unseen_generators": list(UNSEEN_GENERATORS),
        "seen_generators": list(SEEN_GENERATORS),
        "seed": SEED,
        "target_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO},
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("Split statistics written: %s", STATS_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
