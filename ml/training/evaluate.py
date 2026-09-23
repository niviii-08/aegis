"""Evaluation metrics for forgetting-risk classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def select_threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick decision threshold on validation data (no test peeking)."""
    if len(y_true) == 0:
        return 0.5
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t
