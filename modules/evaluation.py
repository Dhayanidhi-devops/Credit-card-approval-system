"""Model evaluation metrics and chart helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


FINTECH_COLORS = {
    "navy": "#0B1F3A",
    "blue": "#1F6FEB",
    "light_blue": "#58A6FF",
    "teal": "#2DD4BF",
    "white": "#F8FAFC",
    "muted": "#94A3B8",
    "approved": "#10B981",
    "declined": "#EF4444",
}


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(str)
    y_pred_arr = np.asarray(y_pred).astype(str)
    unique_labels = labels or sorted(set(y_true_arr) | set(y_pred_arr))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_weighted": float(
            precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "labels": unique_labels,
        "confusion_matrix": confusion_matrix(
            y_true_arr, y_pred_arr, labels=unique_labels
        ).tolist(),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=unique_labels,
            zero_division=0,
            output_dict=True,
        ),
        "classification_report_text": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=unique_labels,
            zero_division=0,
        ),
    }

    metrics["roc_auc"] = None
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba))
            elif y_proba.shape[1] == 2:
                # Prefer Approved class if present.
                pos_index = 1
                if "Approved" in unique_labels:
                    pos_index = list(unique_labels).index("Approved") if "Approved" in unique_labels else 1
                    # Align with estimator class order handled by caller; use last column as positive by default.
                    pos_index = 1
                metrics["roc_auc"] = float(roc_auc_score(
                    (y_true_arr == "Approved").astype(int)
                    if "Approved" in unique_labels
                    else pd.get_dummies(y_true_arr).iloc[:, -1],
                    y_proba[:, pos_index],
                ))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true_arr, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def metrics_to_row(model_name: str, split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model_name,
        "split": split_name,
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "roc_auc": metrics.get("roc_auc"),
    }


def select_best_model(
    results: dict[str, dict[str, Any]],
    metric: str = "macro_f1",
    prefer_balanced: bool = True,
) -> tuple[str | None, float | None]:
    """Pick the best model using the selected metric, with imbalance-aware default."""
    if not results:
        return None, None

    chosen_metric = metric
    if prefer_balanced and metric == "accuracy":
        # Soft guidance: if caller insists on accuracy we still honor it,
        # but prefer_balanced is used by UI defaults (macro_f1).
        chosen_metric = metric

    best_name = None
    best_score = float("-inf")
    for name, payload in results.items():
        test_metrics = payload.get("test_metrics", {})
        score = test_metrics.get(chosen_metric)
        if score is None:
            continue
        if float(score) > best_score:
            best_score = float(score)
            best_name = name
    return best_name, (None if best_name is None else best_score)


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], title: str) -> go.Figure:
    fig = px.imshow(
        np.array(cm),
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale=["#0B1F3A", "#1F6FEB", "#58A6FF"],
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    positive_label: str = "Approved",
    title: str = "ROC Curve",
) -> go.Figure | None:
    try:
        y_bin = (np.asarray(y_true).astype(str) == positive_label).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba)
        auc = roc_auc_score(y_bin, y_proba)
    except Exception:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={auc:.3f})",
            line=dict(color=FINTECH_COLORS["light_blue"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(color=FINTECH_COLORS["muted"], dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    positive_label: str = "Approved",
    title: str = "Precision-Recall Curve",
) -> go.Figure | None:
    try:
        y_bin = (np.asarray(y_true).astype(str) == positive_label).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_proba)
    except Exception:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="PR Curve",
            line=dict(color=FINTECH_COLORS["teal"], width=3),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_metric_comparison(summary_df: pd.DataFrame, metric: str = "macro_f1") -> go.Figure:
    fig = px.bar(
        summary_df,
        x="model",
        y=metric,
        color="model",
        title=f"Model Comparison — {metric.replace('_', ' ').title()}",
        color_discrete_sequence=["#1F6FEB", "#58A6FF", "#2DD4BF", "#93C5FD"],
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
        yaxis_title=metric.replace("_", " ").title(),
        xaxis_title="",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def class_imbalance_ratio(target_counts: dict[str, int]) -> float:
    if not target_counts:
        return 1.0
    values = [v for v in target_counts.values() if v > 0]
    if len(values) < 2:
        return 1.0
    return float(max(values) / min(values))