"""Inference service — loads trained artifacts only (no training on predict)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.inference.loader import load_model_bundle


class ForgettingPredictionError(ValueError):
    """Invalid input for forgetting-risk inference."""


class ForgettingPredictionService:
    def __init__(self, artifact_dir: Path | None = None) -> None:
        self._bundle = load_model_bundle(artifact_dir)
        self._estimator = self._bundle["estimator"]
        self._feature_columns: list[str] = list(self._bundle["feature_columns"])
        self._threshold = float(self._bundle["threshold"])
        self._metadata: dict[str, Any] = dict(self._bundle["metadata"])

    @property
    def model_version(self) -> str:
        return str(self._metadata.get("dataset_version", "unknown"))

    @property
    def feature_schema_version(self) -> str:
        return str(self._metadata.get("feature_schema_version", "unknown"))

    def required_features(self) -> list[str]:
        return list(self._feature_columns)

    def predict_one(self, features: dict[str, Any]) -> dict[str, Any]:
        frame = self._validate_and_frame(features)
        prob = float(self._estimator.predict_proba(frame)[0, 1])
        prob = float(np.clip(prob, 0.0, 1.0))
        label = int(prob >= self._threshold)
        return {
            "will_forget": bool(label),
            "forgetting_probability": prob,
            "threshold": self._threshold,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
        }

    def predict_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict_one(row) for row in rows]

    def _validate_and_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        if not isinstance(features, dict):
            raise ForgettingPredictionError("Features must be a JSON object / dict.")

        missing = [c for c in self._feature_columns if c not in features]
        if missing:
            raise ForgettingPredictionError(
                f"Missing required features: {missing}. "
                f"Expected: {self._feature_columns}"
            )

        unknown = [k for k in features if k not in self._feature_columns]
        if unknown:
            raise ForgettingPredictionError(f"Unexpected feature keys: {unknown}")

        row = {col: features[col] for col in self._feature_columns}
        self._validate_types(row)
        return pd.DataFrame([row])

    @staticmethod
    def _validate_types(row: dict[str, Any]) -> None:
        numeric_keys = {
            "interval_since_last_review_hours",
            "log_interval_since_last_review",
            "cumulative_reviews",
            "rolling_accuracy_5",
            "rolling_accuracy_10",
            "item_difficulty",
            "session_index",
            "hour_of_day_sin",
            "hour_of_day_cos",
            "day_of_week_sin",
            "day_of_week_cos",
        }
        for key in numeric_keys:
            if key not in row:
                continue
            try:
                float(row[key])
            except (TypeError, ValueError) as exc:
                raise ForgettingPredictionError(f"Feature '{key}' must be numeric.") from exc

        if "modality_tag" in row and row["modality_tag"] not in {"image", "audio", "video"}:
            raise ForgettingPredictionError(
                "modality_tag must be one of: image, audio, video"
            )
