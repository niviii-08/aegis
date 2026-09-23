"""Forgetting-risk tabular inference (loads ML artifacts, no training)."""

from __future__ import annotations

import logging
from pathlib import Path

from ml.inference.service import ForgettingPredictionService

logger = logging.getLogger(__name__)


class ForgettingInferenceService:
    def __init__(self, artifact_dir: Path | None = None) -> None:
        self._inner: ForgettingPredictionService | None = None
        self._artifact_dir = artifact_dir
        self._load()

    def _load(self) -> None:
        try:
            self._inner = ForgettingPredictionService(self._artifact_dir)
            logger.info("Forgetting-risk model loaded from artifacts.")
        except FileNotFoundError as exc:
            logger.warning("Forgetting model not available: %s", exc)
            self._inner = None

    @property
    def available(self) -> bool:
        return self._inner is not None

    def predict(self, features: dict) -> dict:
        if self._inner is None:
            raise RuntimeError(
                "Forgetting model not loaded. Train with: python -m ml.training.train"
            )
        return self._inner.predict_one(features)

    def model_info(self) -> dict:
        if self._inner is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "model_version": self._inner.model_version,
            "feature_schema_version": self._inner.feature_schema_version,
            "required_features": self._inner.required_features(),
        }
