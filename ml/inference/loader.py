"""Load persisted forgetting-risk model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def default_artifact_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "forgotting_risk"


def load_model_bundle(artifact_dir: Path | None = None) -> dict[str, Any]:
    base = artifact_dir or default_artifact_dir()
    model_path = base / "forgetting_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run: python -m ml.training.train"
        )
    bundle = joblib.load(model_path)
    required = {"estimator", "feature_columns", "threshold", "metadata"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Corrupt model bundle; missing keys: {sorted(missing)}")
    return bundle
