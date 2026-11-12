"""Model explainability helpers (permutation importance, tree importance, optional SHAP)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from modules.config import RANDOM_SEED

SHAP_AVAILABLE = False
try:
    import shap  # noqa: F401

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def _transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps.get("preprocess")
    if preprocessor is None:
        return []
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def compute_permutation_importance(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 8,
    max_samples: int = 400,
) -> pd.DataFrame:
    """Permutation importance on original feature columns (model association, not causation)."""
    if len(x) > max_samples:
        sample = x.sample(n=max_samples, random_state=RANDOM_SEED)
        y_sample = y.loc[sample.index]
    else:
        sample = x
        y_sample = y

    result = permutation_importance(
        pipeline,
        sample,
        y_sample,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="f1_macro",
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {
            "feature": list(sample.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return frame.reset_index(drop=True)


def compute_tree_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    names = _transformed_feature_names(pipeline)
    importances = np.asarray(model.feature_importances_)
    if not len(names) or len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]
    frame = pd.DataFrame({"feature": names, "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_shap_summary(
    pipeline: Pipeline,
    x: pd.DataFrame,
    max_samples: int = 100,
) -> dict[str, Any] | None:
    """Optional SHAP explanation. Returns None if SHAP is unavailable or fails."""
    if not SHAP_AVAILABLE:
        return None
    try:
        import shap

        sample = x.sample(n=min(max_samples, len(x)), random_state=RANDOM_SEED)
        # Explain model on transformed features when possible.
        preprocess = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        transformed = preprocess.transform(sample)
        feature_names = _transformed_feature_names(pipeline)
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, transformed)
            shap_values = explainer(transformed)
        else:
            explainer = shap.Explainer(model.predict, transformed)
            shap_values = explainer(transformed)

        # Aggregate mean |SHAP| for global ranking.
        values = getattr(shap_values, "values", None)
        if values is None:
            return None
        arr = np.asarray(values)
        if arr.ndim == 3:
            # class, sample, feature -> use positive class if binary else mean abs
            arr = arr[:, :, -1] if arr.shape[0] <= arr.shape[-1] else arr[:, -1, :]
            # Handle both (samples, features, classes) and (classes, samples, features)
            if arr.ndim == 3:
                arr = np.abs(arr).mean(axis=0)
        mean_abs = np.abs(arr).mean(axis=0)
        if not feature_names or len(feature_names) != len(mean_abs):
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]
        frame = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        return {
            "summary": frame.reset_index(drop=True),
            "note": (
                "SHAP values estimate how features associate with the model output "
                "for this synthetic academic demo. They do not prove causation."
            ),
        }
    except Exception:
        return None


def plot_importance_bar(
    frame: pd.DataFrame,
    feature_col: str = "feature",
    value_col: str = "importance_mean",
    title: str = "Feature Importance",
    top_n: int = 15,
):
    data = frame.head(top_n).iloc[::-1]
    fig = px.bar(
        data,
        x=value_col,
        y=feature_col,
        orientation="h",
        title=title,
        color_discrete_sequence=["#58A6FF"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title="Relative influence (association, not causation)",
        yaxis_title="",
    )
    return fig


def explain_prediction_locally(
    pipeline: Pipeline,
    applicant_row: pd.DataFrame,
    background: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Best-effort local explanation for a single applicant."""
    # Prefer tree importances mapped loosely; use SHAP when available.
    shap_result = None
    if background is not None and len(background) > 0:
        try:
            combined = pd.concat([background.sample(n=min(50, len(background)), random_state=RANDOM_SEED), applicant_row], ignore_index=True)
            shap_result = compute_shap_summary(pipeline, combined, max_samples=len(combined))
        except Exception:
            shap_result = None

    tree_imp = compute_tree_feature_importance(pipeline)
    return {
        "shap_available": SHAP_AVAILABLE and shap_result is not None,
        "shap": shap_result,
        "tree_importance": tree_imp,
        "caveat": (
            "Feature influence scores reflect model associations on synthetic data. "
            "They must not be interpreted as causal reasons for approval or decline."
        ),
    }