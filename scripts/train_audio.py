"""Train the AEGIS audio baseline model.

Wrapper that calls src.audio.training.train with configs/audio_baseline.yaml.

Usage::

    python scripts/train_audio.py

Produces: models/audio/baseline_best.pt
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
    config_path = PROJECT_ROOT / "configs" / "audio_baseline.yaml"

    # Verify prerequisites
    splits_dir = PROJECT_ROOT / "data" / "processed" / "audio" / "splits"
    for split in ("train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"):
        split_path = splits_dir / split
        if not split_path.is_file():
            logger.error("Missing audio split: %s", split_path)
            return 1

    metadata_path = PROJECT_ROOT / "data" / "processed" / "audio" / "preprocessing" / "metadata.csv"
    if not metadata_path.is_file():
        logger.error("Missing preprocessing metadata: %s", metadata_path)
        logger.error("Run audio preprocessing first.")
        return 1

    logger.info("=== Training Audio Baseline Model ===")
    logger.info("Config: %s", config_path)

    from audio.training.utils import find_project_root as _find_root, load_training_config
    from audio.training.train import train

    config = load_training_config(config_path, project_root=PROJECT_ROOT)
    checkpoint_path = train(config)

    logger.info("=== Audio Training Complete ===")
    logger.info("Checkpoint: %s", checkpoint_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
