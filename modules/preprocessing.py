"""Feature preprocessing utilities using sklearn Pipeline components."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.config import APPROVED_LABELS, DECLINED_LABELS


def normalize_target_labels(y: pd.Series | Iterable[str]) -> pd.Series:
    """Map common approval labels to Approved / Declined for consistent reporting."""
    series = pd.Series(y).astype(str).str.strip()
    mapped = []
    for value in series:
        lowered = value.lower()
        if lowered in APPROVED_LABELS:
            mapped.append("Approved")
        elif lowered in DECLINED_LABELS:
            mapped.append("Declined")
        else:
            # Preserve original label for multi-class / custom datasets.
            mapped.append(value)
    return pd.Series(mapped, index=series.index, name=getattr(y, "name", "target"))


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a ColumnTransformer that fits only on training data when used in a Pipeline."""
    transformers = []

    if numeric_features:
        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=numeric_steps)
        transformers.append(("num", numeric_pipeline, list(numeric_features)))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, list(categorical_features)))

    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def infer_feature_lists(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical = [c for c in x.columns if c not in numeric]
    return numeric, categorical


def safe_numeric_bounds(series: pd.Series, fallback_min: float, fallback_max: float) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float(fallback_min), float(fallback_max)
    low = float(clean.min())
    high = float(clean.max())
    if np.isclose(low, high):
        high = low + 1.0
    return low, high