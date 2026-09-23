"""End-to-end training pipeline for forgetting-risk prediction."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ml.data.generator import DatasetConfig, generate_forgetting_dataset, save_dataset
from ml.data.leakage import assert_no_target_leakage, assert_temporal_integrity
from ml.data.splits import time_aware_split
from ml.features.build import build_feature_matrix, get_feature_names
from ml.features.schema import feature_metadata
from ml.training.evaluate import compute_metrics, select_threshold_max_f1
from ml.training.models import build_baseline_pipeline, build_xgb_pipeline
from ml.training.report import write_performance_report

logger = logging.getLogger(__name__)

RECOMMENDED_METRIC = "PR-AUC (Average Precision)"
METRIC_RATIONALE = (
    "Forgetting-risk prediction is a cost-sensitive screening problem: missing a learner "
    "who will forget (false negative) is usually worse than a false alarm. PR-AUC focuses "
    "on ranking positive (high-risk) cases under class imbalance and aligns with recall-focused "
    "interventions. ROC-AUC is reported for completeness but can look optimistic when negatives dominate."
)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_time_series_cv(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
) -> dict[str, Any]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float | None]] = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(x), start=1):
        x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        pipeline.fit(x_tr, y_tr)
        prob = pipeline.predict_proba(x_va)[:, 1]
        m = compute_metrics(y_va.to_numpy(), prob)
        fold_metrics.append(
            {
                "fold": fold,
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "f1": m.get("f1"),
            }
        )

    def _mean(key: str) -> float | None:
        vals = [f[key] for f in fold_metrics if f[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "folds": fold_metrics,
        "roc_auc_mean": _mean("roc_auc"),
        "pr_auc_mean": _mean("pr_auc"),
        "f1_mean": _mean("f1"),
    }


def extract_feature_importance(
    model_pipeline: Pipeline,
    feature_names: list[str],
) -> dict[str, float]:
    clf = model_pipeline.named_steps["model"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).ravel()
    else:
        return {}

    if len(feature_names) != len(importances):
        feature_names = [f"f{i}" for i in range(len(importances))]

    pairs = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda p: p[1],
        reverse=True,
    )
    return {name: float(val) for name, val in pairs}


def compute_shap_artifacts(
    model_bundle: Pipeline,
    x_background: pd.DataFrame,
    x_example: pd.DataFrame,
    artifact_dir: Path,
    max_background: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed; skipping explainability artifacts.")
        return {"enabled": False, "reason": "shap not installed"}

    preprocess = model_bundle.named_steps["preprocess"]
    model = model_bundle.named_steps["model"]
    bg = x_background.sample(n=min(len(x_background), max_background), random_state=42)
    x_bg = preprocess.transform(bg)
    x_ex = preprocess.transform(x_example)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_bg)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_names = get_feature_names(preprocess)
    if len(feature_names) != len(mean_abs):
        feature_names = [f"f{i}" for i in range(len(mean_abs))]

    global_importance = {
        name: float(val)
        for name, val in sorted(
            zip(feature_names, mean_abs, strict=False),
            key=lambda p: p[1],
            reverse=True,
        )
    }

    single = explainer.shap_values(x_ex)
    if isinstance(single, list):
        single = single[1]
    individual = {
        "feature_names": feature_names,
        "shap_values": single[0].tolist(),
        "expected_value": float(explainer.expected_value)
        if not isinstance(explainer.expected_value, (list, np.ndarray))
        else float(np.asarray(explainer.expected_value).ravel()[0]),
    }

    shap_dir = artifact_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "shap_global_importance.json").write_text(
        json.dumps(global_importance, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "shap_individual_example.json").write_text(
        json.dumps(individual, indent=2),
        encoding="utf-8",
    )
    return {"enabled": True, "global_top_features": list(global_importance.keys())[:10]}


def train_pipeline(config_path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    cfg_path = config_path or (root / "ml" / "config" / "default.yaml")
    cfg = load_config(cfg_path)

    seed = int(cfg["seed"])
    np.random.seed(seed)

    data_dir = root / cfg["paths"]["data_dir"]
    artifact_dir = root / cfg["paths"]["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = DatasetConfig(
        n_users=int(cfg["dataset"]["n_users"]),
        n_items=int(cfg["dataset"]["n_items"]),
        n_days=int(cfg["dataset"]["n_days"]),
        reviews_per_user_mean=int(cfg["dataset"]["reviews_per_user_mean"]),
        retention_horizon_days=int(cfg["dataset"]["retention_horizon_days"]),
        random_seed=int(cfg["dataset"]["random_seed"]),
    )
    if cfg["training"].get("quick_mode"):
        ds_cfg = DatasetConfig(
            n_users=40,
            n_items=20,
            n_days=60,
            reviews_per_user_mean=20,
            retention_horizon_days=ds_cfg.retention_horizon_days,
            random_seed=ds_cfg.random_seed,
        )

    events = generate_forgetting_dataset(ds_cfg)
    save_dataset(events, data_dir, cfg["dataset_version"], seed)

    numeric = list(cfg["features"]["numeric"])
    categorical = list(cfg["features"]["categorical"])
    forbidden = list(cfg["features"]["forbidden"])
    assert_no_target_leakage(numeric + categorical, forbidden)

    splits = time_aware_split(
        events,
        train_frac=float(cfg["splits"]["train_frac"]),
        val_frac=float(cfg["splits"]["val_frac"]),
        test_frac=float(cfg["splits"]["test_frac"]),
    )
    assert_temporal_integrity(splits.train, splits.val, splits.test)

    x_train, y_train = build_feature_matrix(
        splits.train, numeric, categorical, forbidden
    )
    x_val, y_val = build_feature_matrix(splits.val, numeric, categorical, forbidden)
    x_test, y_test = build_feature_matrix(splits.test, numeric, categorical, forbidden)

    quick = bool(cfg["training"].get("quick_mode"))
    baseline = build_baseline_pipeline(numeric, categorical, cfg["models"]["baseline"]["params"])
    baseline.fit(x_train, y_train)

    xgb = build_xgb_pipeline(
        numeric,
        categorical,
        cfg["models"]["xgboost"]["params"],
        y_train.to_numpy(),
        quick_mode=quick,
    )

    cv_folds = int(cfg["training"]["cv_folds"])
    cv_summary = run_time_series_cv(xgb, x_train, y_train, n_splits=cv_folds)

    xgb.fit(x_train, y_train)

    calibrated: Any = xgb
    if cfg["training"].get("calibrate", True):
        method = str(cfg["training"].get("calibration_method", "isotonic"))
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(FrozenEstimator(xgb), method=method)
        except ImportError:
            calibrated = CalibratedClassifierCV(xgb, method=method, cv="prefit")
        calibrated.fit(x_val, y_val)

    test_prob = calibrated.predict_proba(x_test)[:, 1]
    val_prob = calibrated.predict_proba(x_val)[:, 1]
    baseline_test_prob = baseline.predict_proba(x_test)[:, 1]

    threshold = select_threshold_max_f1(y_val.to_numpy(), val_prob)
    primary_metrics = compute_metrics(y_test.to_numpy(), test_prob, threshold)
    val_metrics = compute_metrics(y_val.to_numpy(), val_prob, threshold)
    baseline_metrics = compute_metrics(y_test.to_numpy(), baseline_test_prob, threshold)

    preprocess_fitted = xgb.named_steps["preprocess"]
    feature_names = get_feature_names(preprocess_fitted)
    importance = extract_feature_importance(xgb, feature_names)

    meta = feature_metadata(numeric, categorical, cfg["dataset_version"], seed)
    meta["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["config_path"] = str(cfg_path)

    bundle = {
        "model_version": cfg["dataset_version"],
        "feature_columns": numeric + categorical,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "forbidden_columns": forbidden,
        "threshold": threshold,
        "estimator": calibrated,
        "raw_xgb_pipeline": xgb,
        "baseline_pipeline": baseline,
        "metadata": meta,
    }

    model_path = artifact_dir / "forgetting_model.joblib"
    joblib.dump(bundle, model_path)

    metrics_payload = {
        "primary_model": cfg["training"]["primary_model"],
        "validation": val_metrics,
        "test": primary_metrics,
        "baseline_test": baseline_metrics,
        "cross_validation": cv_summary,
        "recommended_metric": RECOMMENDED_METRIC,
        "metric_rationale": METRIC_RATIONALE,
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "feature_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "feature_importance.json").write_text(
        json.dumps(importance, indent=2),
        encoding="utf-8",
    )

    shap_info = compute_shap_artifacts(
        xgb,
        x_train,
        x_test.iloc[[int(cfg["shap"]["individual_example_index"])]],
        artifact_dir,
        max_background=int(cfg["shap"]["max_background_samples"]),
        enabled=bool(cfg["shap"]["enabled"]) and not quick,
    )
    (artifact_dir / "shap_status.json").write_text(
        json.dumps(shap_info, indent=2),
        encoding="utf-8",
    )

    write_performance_report(
        artifact_dir / "performance_report.md",
        primary_metrics,
        baseline_metrics,
        cv_summary,
        RECOMMENDED_METRIC,
        METRIC_RATIONALE,
    )
    shutil.copy2(cfg_path, artifact_dir / "training_config.yaml")

    logger.info("Training complete. Artifacts: %s", artifact_dir)
    return artifact_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train forgetting-risk prediction model")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: ml/config/default.yaml)",
    )
    args = parser.parse_args()
    train_pipeline(args.config)


if __name__ == "__main__":
    main()
