"""Feature matrix construction and sklearn preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data.leakage import assert_no_target_leakage


def build_feature_matrix(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    forbidden: list[str],
    target_col: str = "will_forget",
) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = numeric_features + categorical_features
    assert_no_target_leakage(feature_columns, forbidden + [target_col, "recall_success_at_t"])

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns in dataframe: {missing_cols}")

    x = df[feature_columns].copy()
    y = df[target_col].astype(int)
    return x, y


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def row_to_model_input(row: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    return {col: row[col] for col in feature_columns if col in row}
