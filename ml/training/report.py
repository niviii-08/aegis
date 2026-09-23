"""Human-readable performance report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_performance_report(
    path: Path,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    cv_summary: dict[str, Any],
    recommended_metric: str,
    metric_rationale: str,
) -> None:
    lines = [
        "# Forgetting-Risk Model — Performance Report",
        "",
        "## Recommended decision metric",
        f"**{recommended_metric}** — {metric_rationale}",
        "",
        "## Primary model (test set)",
        "",
        _metrics_table(metrics),
        "",
        "## Baseline logistic regression (test set)",
        "",
        _metrics_table(baseline_metrics),
        "",
        "## Cross-validation (time-series folds on training data)",
        "",
        f"- Mean ROC-AUC: {cv_summary.get('roc_auc_mean')}",
        f"- Mean PR-AUC: {cv_summary.get('pr_auc_mean')}",
        f"- Mean F1: {cv_summary.get('f1_mean')}",
        "",
        "## Confusion matrix (primary model, test)",
        "",
        str(metrics.get("confusion_matrix")),
        "",
        "## Notes",
        "- Splits are time-aware on `event_timestamp` (no random shuffle).",
        "- Identifiers and future outcomes are excluded from features to prevent leakage.",
        "- Accuracy alone is misleading under class imbalance; prioritize recall/PR-AUC for risk screening.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _metrics_table(metrics: dict[str, Any]) -> str:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = [f"| {k} | {metrics.get(k)} |" for k in keys]
    return "\n".join(["| metric | value |", "| --- | --- |", *rows])
