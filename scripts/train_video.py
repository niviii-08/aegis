"""Train the AEGIS video baseline model.

Wrapper that calls src.video.training.train with configs/video_baseline.yaml.

Usage::

    python scripts/train_video.py

Produces: models/video/baseline_best.pt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError("Could not locate AEGIS project root")


PROJECT_ROOT = find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "video_baseline.yaml"

    # Verify prerequisites
    splits_dir = PROJECT_ROOT / "data" / "processed" / "video" / "splits"
    for split in ("train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"):
        split_path = splits_dir / split
        if not split_path.is_file():
            logger.error("Missing video split: %s", split_path)
            logger.error("Run scripts/extract_video_frames.py and scripts/generate_video_splits.py first.")
            return 1

    frames_root = PROJECT_ROOT / "data" / "processed" / "video" / "frames"
    if not frames_root.is_dir() or not any(frames_root.iterdir()):
        logger.error("Video frames not found at: %s", frames_root)
        logger.error("Run scripts/extract_video_frames.py first.")
        return 1

    logger.info("=== Training Video Baseline Model ===")
    logger.info("Config: %s", config_path)

    from video.training.utils import load_training_config
    from video.training.train import train

    config = load_training_config(config_path, project_root=PROJECT_ROOT)
    checkpoint_path = train(config)

    logger.info("=== Video Training Complete ===")
    logger.info("Checkpoint: %s", checkpoint_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
