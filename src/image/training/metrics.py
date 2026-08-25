"""Classification metrics for image baseline evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "balanced_accuracy",
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary classification metrics for P(fake) predictions."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.size == 0:
        return _empty_metrics()

    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "support": int(y_true.size),
        "threshold": threshold,
    }

    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = None
        metrics["roc_auc_note"] = "undefined_single_class"
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def _empty_metrics() -> dict[str, Any]:
    return {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "balanced_accuracy": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "support": 0,
        "note": "empty_split",
    }


def compute_generalization_gap(
    seen_metrics: dict[str, Any],
    unseen_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compute seen_test_metric - unseen_test_metric for main metrics."""
    gap: dict[str, Any] = {}
    notes: list[str] = []

    unseen_support = unseen_metrics.get("support", 0)
    if unseen_support == 0:
        notes.append("test_unseen is empty; generalization_gap is not computable.")

    for name in METRIC_NAMES:
        seen_value = seen_metrics.get(name)
        unseen_value = unseen_metrics.get(name)
        if (
            unseen_support == 0
            or seen_value is None
            or unseen_value is None
        ):
            gap[name] = None
        else:
            gap[name] = round(float(seen_value) - float(unseen_value), 6)

    return {"metrics": gap, "notes": notes}
