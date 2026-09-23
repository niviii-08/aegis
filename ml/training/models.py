"""Baseline and XGBoost estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml.features.build import build_preprocessor


def _pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos == 0:
        return 1.0
    return max(neg / pos, 1.0)


def build_baseline_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict[str, Any],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    clf = LogisticRegression(
        max_iter=int(params.get("max_iter", 2000)),
        class_weight=params.get("class_weight", "balanced"),
        C=float(params.get("C", 1.0)),
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def build_xgb_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict[str, Any],
    y_train: np.ndarray,
    quick_mode: bool = False,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    xgb_params = dict(params)
    scale = xgb_params.pop("scale_pos_weight", "auto")
    if scale == "auto":
        scale = _pos_weight(y_train)

    n_estimators = int(xgb_params.get("n_estimators", 300))
    if quick_mode:
        n_estimators = min(n_estimators, 30)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=int(xgb_params.get("max_depth", 5)),
        learning_rate=float(xgb_params.get("learning_rate", 0.05)),
        subsample=float(xgb_params.get("subsample", 0.85)),
        colsample_bytree=float(xgb_params.get("colsample_bytree", 0.85)),
        min_child_weight=float(xgb_params.get("min_child_weight", 3)),
        reg_lambda=float(xgb_params.get("reg_lambda", 1.0)),
        scale_pos_weight=float(scale),
        eval_metric=str(xgb_params.get("eval_metric", "logloss")),
        random_state=int(xgb_params.get("random_state", 42)),
        n_jobs=int(xgb_params.get("n_jobs", -1)),
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])
