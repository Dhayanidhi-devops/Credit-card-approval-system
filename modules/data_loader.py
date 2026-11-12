"""Dataset loading, validation, and profiling utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from modules.config import (
    DEFAULT_DATASET_PATH,
    FORBIDDEN_FEATURE_KEYWORDS,
    ID_CANDIDATES,
    TARGET_CANDIDATES,
)


@dataclass
class DatasetProfile:
    """Structured summary of a loaded dataset."""

    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    duplicate_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    target_column: str | None
    id_column: str | None
    target_distribution: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source_name: str = "unknown"


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be used for training or prediction."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    return cleaned


def identify_target_column(columns: list[str], preferred: str | None = None) -> str | None:
    if preferred and preferred in columns:
        return preferred
    lower_map = {c.lower(): c for c in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def identify_id_column(columns: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in ID_CANDIDATES:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def is_forbidden_feature(column_name: str) -> bool:
    lowered = column_name.lower().replace(" ", "_")
    return any(keyword in lowered for keyword in FORBIDDEN_FEATURE_KEYWORDS)


def classify_columns(
    df: pd.DataFrame,
    target_column: str | None = None,
    id_column: str | None = None,
) -> tuple[list[str], list[str]]:
    """Identify usable numeric and categorical predictor columns."""
    exclude = {c for c in (target_column, id_column) if c}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in exclude or is_forbidden_feature(col):
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            # Treat low-cardinality integer columns as categorical when appropriate.
            nunique = series.nunique(dropna=True)
            if pd.api.types.is_integer_dtype(series) and nunique <= 2 and nunique > 0:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def load_dataframe_from_path(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise DatasetValidationError(
            f"Unsupported file type '{suffix}'. Please upload an Excel (.xlsx) or CSV (.csv) file."
        )
    return _normalize_columns(df)


def load_dataframe_from_upload(uploaded_file: BinaryIO, filename: str) -> pd.DataFrame:
    name = filename.lower()
    raw = uploaded_file.read()
    buffer = BytesIO(raw)
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buffer)
    elif name.endswith(".csv"):
        df = pd.read_csv(buffer)
    else:
        raise DatasetValidationError(
            "Unsupported upload type. Please upload an Excel (.xlsx/.xls) or CSV (.csv) file."
        )
    return _normalize_columns(df)


def load_default_dataset() -> pd.DataFrame:
    return load_dataframe_from_path(DEFAULT_DATASET_PATH)


def profile_dataset(
    df: pd.DataFrame,
    source_name: str = "dataset",
    preferred_target: str | None = None,
) -> DatasetProfile:
    if df is None or df.empty:
        raise DatasetValidationError("The dataset is empty. Please provide a non-empty Excel/CSV file.")

    working = _normalize_columns(df)
    target = identify_target_column(list(working.columns), preferred_target)
    id_col = identify_id_column(list(working.columns))
    numeric_cols, categorical_cols = classify_columns(working, target, id_col)

    missing = working.isna().sum()
    missing_counts = {col: int(count) for col, count in missing.items() if count > 0}
    duplicate_count = int(working.duplicated().sum())

    target_distribution: dict[str, int] = {}
    warnings: list[str] = []
    if target is None:
        warnings.append(
            "No target column detected. Expected one of: "
            + ", ".join(TARGET_CANDIDATES)
        )
    else:
        target_distribution = {
            str(k): int(v) for k, v in working[target].value_counts(dropna=False).items()
        }
        if working[target].nunique(dropna=True) < 2:
            warnings.append(
                f"Target column '{target}' has fewer than 2 classes and cannot be used for classification."
            )

    forbidden = [c for c in working.columns if is_forbidden_feature(c)]
    if forbidden:
        warnings.append(
            "Sensitive demographic columns detected and will be excluded from modeling: "
            + ", ".join(forbidden)
        )

    if not numeric_cols and not categorical_cols:
        warnings.append("No usable predictor columns were found after excluding IDs and sensitive fields.")

    return DatasetProfile(
        n_rows=int(working.shape[0]),
        n_cols=int(working.shape[1]),
        columns=list(working.columns),
        dtypes={c: str(working[c].dtype) for c in working.columns},
        missing_counts=missing_counts,
        duplicate_count=duplicate_count,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        target_column=target,
        id_column=id_col,
        target_distribution=target_distribution,
        warnings=warnings,
        source_name=source_name,
    )


def validate_for_training(df: pd.DataFrame, profile: DatasetProfile | None = None) -> DatasetProfile:
    profile = profile or profile_dataset(df)
    if profile.target_column is None:
        raise DatasetValidationError(
            "Required target column is missing. Expected a column such as 'Approval_Status'."
        )
    if profile.target_column not in df.columns:
        raise DatasetValidationError(
            f"Required target column '{profile.target_column}' is missing from the dataset."
        )
    if df[profile.target_column].nunique(dropna=True) < 2:
        raise DatasetValidationError(
            f"Target column '{profile.target_column}' must contain at least two classes."
        )
    if not profile.numeric_columns and not profile.categorical_columns:
        raise DatasetValidationError(
            "No predictor features available for training after excluding IDs and sensitive attributes."
        )
    if profile.n_rows < 20:
        raise DatasetValidationError(
            "Dataset is too small for reliable train/test evaluation (minimum 20 rows recommended)."
        )
    return profile


def get_feature_frame(df: pd.DataFrame, profile: DatasetProfile) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictor matrix X and target series y without mutating the original frame."""
    validate_for_training(df, profile)
    drop_cols = [c for c in (profile.target_column, profile.id_column) if c]
    drop_cols.extend([c for c in df.columns if is_forbidden_feature(c)])
    feature_cols = [c for c in df.columns if c not in set(drop_cols)]
    x = df[feature_cols].copy()
    y = df[profile.target_column].astype(str).copy()
    return x, y


def dataset_summary_dict(profile: DatasetProfile) -> dict[str, Any]:
    return {
        "source_name": profile.source_name,
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "target_column": profile.target_column,
        "id_column": profile.id_column,
        "numeric_columns": profile.numeric_columns,
        "categorical_columns": profile.categorical_columns,
        "missing_counts": profile.missing_counts,
        "duplicate_count": profile.duplicate_count,
        "target_distribution": profile.target_distribution,
        "warnings": profile.warnings,
    }