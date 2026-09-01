#!/usr/bin/env python3
"""Reproducible training entry point for AUDIO modality.

This script provides a single command to train the AEGIS audio baseline model
with full reproducibility and experiment tracking.

Usage:
    python experiments/train_audio.py [--config CONFIG_PATH] [--project-root PROJECT_ROOT]

Example:
    python experiments/train_audio.py --config configs/audio_baseline.yaml
"""

import sys
from pathlib import Path

# Add src to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from audio.training.train import main

if __name__ == "__main__":
    sys.exit(main())