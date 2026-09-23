"""Tests for forgetting-risk ML pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.data.generator import DatasetConfig, generate_forgetting_dataset
from ml.data.leakage import assert_no_target_leakage, assert_temporal_integrity
from ml.data.splits import time_aware_split
from ml.features.build import build_feature_matrix
from ml.inference.loader import load_model_bundle
from ml.inference.service import ForgettingPredictionError, ForgettingPredictionService
from ml.training.train import train_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUICK_CONFIG = PROJECT_ROOT / "ml" / "config" / "quick.yaml"


@pytest.fixture(scope="module")
def trained_artifacts() -> Path:
    return train_pipeline(QUICK_CONFIG)


@pytest.fixture(scope="module")
def sample_features(trained_artifacts: Path) -> dict:
    bundle = load_model_bundle(trained_artifacts)
    cols = bundle["feature_columns"]
    return {
        "interval_since_last_review_hours": 48.0,
        "log_interval_since_last_review": float(np.log1p(48.0)),
        "cumulative_reviews": 10,
        "rolling_accuracy_5": 0.6,
        "rolling_accuracy_10": 0.55,
        "item_difficulty": 0.4,
        "session_index": 9,
        "hour_of_day_sin": 0.5,
        "hour_of_day_cos": 0.86,
        "day_of_week_sin": 0.0,
        "day_of_week_cos": 1.0,
        "modality_tag": "image",
    }


def test_feature_generation_reproducible():
    cfg = DatasetConfig(n_users=10, n_items=5, reviews_per_user_mean=8, random_seed=99)
    a = generate_forgetting_dataset(cfg)
    b = generate_forgetting_dataset(cfg)
    assert len(a) == len(b)
    assert a["will_forget"].tolist() == b["will_forget"].tolist()


def test_time_aware_split_and_leakage():
    cfg = DatasetConfig(n_users=30, n_items=10, reviews_per_user_mean=12, random_seed=1)
    df = generate_forgetting_dataset(cfg)
    splits = time_aware_split(df)
    assert_temporal_integrity(splits.train, splits.val, splits.test)
    numeric = [
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
    ]
    forbidden = [
        "will_forget",
        "user_id",
        "item_id",
        "event_timestamp",
        "next_review_timestamp",
        "future_recall_success",
        "label_horizon_end",
    ]
    x, y = build_feature_matrix(
        splits.train,
        numeric,
        ["modality_tag"],
        forbidden,
    )
    assert_no_target_leakage(x.columns.tolist(), forbidden)
    assert set(y.unique()).issubset({0, 1})


def test_training_writes_artifacts(trained_artifacts: Path):
    required = [
        "forgetting_model.joblib",
        "metrics.json",
        "feature_metadata.json",
        "feature_importance.json",
        "performance_report.md",
        "training_config.yaml",
    ]
    for name in required:
        assert (trained_artifacts / name).exists()


def test_model_loading(trained_artifacts: Path):
    bundle = load_model_bundle(trained_artifacts)
    assert "estimator" in bundle
    assert bundle["feature_columns"]


def test_prediction_probability_range(trained_artifacts: Path, sample_features: dict):
    svc = ForgettingPredictionService(trained_artifacts)
    out = svc.predict_one(sample_features)
    assert 0.0 <= out["forgetting_probability"] <= 1.0
    assert isinstance(out["will_forget"], bool)


def test_missing_features(trained_artifacts: Path):
    svc = ForgettingPredictionService(trained_artifacts)
    with pytest.raises(ForgettingPredictionError, match="Missing required features"):
        svc.predict_one({"interval_since_last_review_hours": 1.0})


def test_invalid_input(trained_artifacts: Path, sample_features: dict):
    svc = ForgettingPredictionService(trained_artifacts)
    bad = dict(sample_features)
    bad["modality_tag"] = "text"
    with pytest.raises(ForgettingPredictionError, match="modality_tag"):
        svc.predict_one(bad)

    with pytest.raises(ForgettingPredictionError, match="JSON object"):
        svc.predict_one([])  # type: ignore[arg-type]


def test_metrics_json_contains_core_metrics(trained_artifacts: Path):
    import json

    metrics = json.loads((trained_artifacts / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"):
        assert key in test
