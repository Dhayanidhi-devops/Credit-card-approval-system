"""Applicant prediction helpers and session history utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from modules.preprocessing import normalize_target_labels


class PredictionValidationError(ValueError):
    """Raised when applicant form input fails validation."""


def validate_applicant_input(
    values: dict[str, Any],
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in values or values[c] in (None, "")]
    if missing:
        raise PredictionValidationError(
            "Required fields missing: " + ", ".join(missing)
        )

    row: dict[str, Any] = {}
    for col in feature_columns:
        value = values[col]
        if col in numeric_features:
            try:
                num = float(value)
            except (TypeError, ValueError) as exc:
                raise PredictionValidationError(f"'{col}' must be a numeric value.") from exc
            if not np.isfinite(num):
                raise PredictionValidationError(f"'{col}' must be a finite number.")
            row[col] = num
        else:
            text = str(value).strip()
            if not text:
                raise PredictionValidationError(f"'{col}' is required.")
            row[col] = text

    return pd.DataFrame([row], columns=feature_columns)


def predict_applicant(
    pipeline: Pipeline,
    applicant_df: pd.DataFrame,
    model_name: str = "model",
) -> dict[str, Any]:
    try:
        pred_raw = pipeline.predict(applicant_df)[0]
    except Exception as exc:
        raise RuntimeError(f"Prediction failed: {exc}") from exc

    pred_label = str(normalize_target_labels([pred_raw]).iloc[0])
    probabilities: dict[str, float] | None = None
    approval_probability = None

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(applicant_df)[0]
            classes = [str(c) for c in pipeline.named_steps["model"].classes_]
            mapped_classes = list(normalize_target_labels(classes))
            probabilities = {
                mapped_classes[i]: float(proba[i]) for i in range(len(mapped_classes))
            }
            approval_probability = probabilities.get("Approved")
            if approval_probability is None and probabilities:
                # Fall back to max class probability label mapping.
                approval_probability = max(probabilities.values())
        except Exception:
            probabilities = None

    explanation = (
        f"The selected model ({model_name}) estimated the applicant outcome as "
        f"**{pred_label}** based on the supplied credit and employment attributes. "
        "This is a statistical association from a synthetic academic dataset, "
        "not a guaranteed real-world approval decision."
    )
    if probabilities:
        pretty = ", ".join(f"{k}: {v:.1%}" for k, v in probabilities.items())
        explanation += f" Class probabilities (model estimates only): {pretty}."

    return {
        "prediction": pred_label,
        "probabilities": probabilities,
        "approval_probability": approval_probability,
        "model_name": model_name,
        "explanation": explanation,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": applicant_df.iloc[0].to_dict(),
    }


def make_history_record(
    prediction_result: dict[str, Any],
    reference: str | None = None,
) -> dict[str, Any]:
    probs = prediction_result.get("probabilities") or {}
    return {
        "reference": reference or f"REF-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "prediction": prediction_result.get("prediction"),
        "approval_probability": prediction_result.get("approval_probability"),
        "probability_approved": probs.get("Approved"),
        "probability_declined": probs.get("Declined"),
        "model": prediction_result.get("model_name"),
        "timestamp": prediction_result.get("timestamp"),
    }


def history_to_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(
            columns=[
                "reference",
                "prediction",
                "approval_probability",
                "probability_approved",
                "probability_declined",
                "model",
                "timestamp",
            ]
        )
    return pd.DataFrame(history)