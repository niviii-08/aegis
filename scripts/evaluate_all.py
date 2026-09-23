"""Evaluate all AEGIS baseline models.

Usage::
    python scripts/evaluate_all.py
"""

from __future__ import annotations

import logging
import subprocess
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
    models = [
        ("Image", "image.training.evaluate", "models/image/baseline_best.pt"),
        ("Audio", "audio.training.evaluate", "models/audio/baseline_best.pt"),
        ("Video", "video.training.evaluate", "models/video/baseline_best.pt"),
    ]
    
    success = True
    for name, module, ckpt in models:
        ckpt_path = PROJECT_ROOT / ckpt
        if not ckpt_path.exists():
            logger.warning(f"[{name}] Checkpoint not found: {ckpt_path}. Skipping.")
            continue
            
        logger.info(f"=== Evaluating {name} Baseline ===")
        cmd = [sys.executable, "-m", module, "--checkpoint", str(ckpt_path)]
        
        try:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
            logger.info(f"[{name}] Evaluation complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"[{name}] Evaluation failed with exit code {e.returncode}")
            success = False
            
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
