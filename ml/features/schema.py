"""Feature schema and version metadata for reproducibility."""

from __future__ import annotations

from typing import Any

FEATURE_SCHEMA_VERSION = "1.0.0"


def feature_metadata(
    numeric_features: list[str],
    categorical_features: list[str],
    dataset_version: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "random_seed": seed,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "will_forget",
        "target_definition": (
            "Binary indicator: recall failure for the same user-item pair "
            "within retention_horizon_days after the review event."
        ),
        "leakage_policy": (
            "Identifiers, timestamps, and post-event outcomes are excluded from model inputs."
        ),
    }
