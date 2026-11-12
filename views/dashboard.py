"""Dashboard page — dynamic KPIs and charts from loaded data and trained models."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.evaluation import FINTECH_COLORS, plot_metric_comparison
from modules.model_training import comparison_frame
from modules.preprocessing import normalize_target_labels
from modules.ui import disclaimer_banner, metric_card, require_dataset, section_title


def _distribution_frame(series: pd.Series, name: str) -> pd.DataFrame:
    counts = series.astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [name, "count"]
    return counts


def render() -> None:
    section_title(
        "Dashboard",
        "Live portfolio overview computed from the currently loaded synthetic dataset and trained models.",
    )
    disclaimer_banner()

    if not require_dataset():
        return

    df = st.session_state.dataset.copy()
    profile = st.session_state.dataset_profile
    target_col = profile.target_column

    if target_col and target_col in df.columns:
        target = normalize_target_labels(df[target_col])
    else:
        target = pd.Series(["Unknown"] * len(df))

    total = len(df)
    approved = int((target == "Approved").sum())
    declined = int((target == "Declined").sum())
    other = total - approved - declined
    approval_pct = (approved / total * 100) if total else 0.0
    decline_pct = (declined / total * 100) if total else 0.0

    bundle = st.session_state.get("training_bundle")
    best_model = "Not trained"
    best_acc = "—"
    best_macro_f1 = "—"
    if bundle and bundle.best_model_name:
        best_key = bundle.best_model_name
        best_model = bundle.results[best_key]["display_name"]
        test_metrics = bundle.results[best_key]["test_metrics"]
        best_acc = f"{test_metrics.get('accuracy', 0) * 100:.2f}%"
        best_macro_f1 = f"{test_metrics.get('macro_f1', 0):.3f}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Applicants", f"{total:,}", profile.source_name)
    with c2:
        metric_card("Approved", f"{approved:,}", f"{approval_pct:.1f}%")
    with c3:
        metric_card("Declined", f"{declined:,}", f"{decline_pct:.1f}%")
    with c4:
        metric_card("Best Model", best_model, f"Accuracy {best_acc} · Macro F1 {best_macro_f1}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Approval %", f"{approval_pct:.1f}%")
    with c6:
        metric_card("Decline %", f"{decline_pct:.1f}%")
    with c7:
        metric_card("Model Accuracy", best_acc)
    with c8:
        metric_card("Macro F1-score", best_macro_f1)

    if other > 0:
        st.caption(f"Note: {other} rows have labels outside Approved/Declined after normalization.")

    left, right = st.columns(2)
    with left:
        dist = _distribution_frame(target, "status")
        fig = px.pie(
            dist,
            names="status",
            values="count",
            title="Approval vs Decline Distribution",
            color="status",
            color_discrete_map={
                "Approved": FINTECH_COLORS["approved"],
                "Declined": FINTECH_COLORS["declined"],
            },
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig, width="stretch")

    with right:
        if "Annual_Income_INR" in df.columns:
            fig = px.histogram(
                df,
                x="Annual_Income_INR",
                nbins=30,
                title="Applicant Income Distribution (INR)",
                color_discrete_sequence=[FINTECH_COLORS["blue"]],
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Annual income column not found in the current dataset.")

    left2, right2 = st.columns(2)
    with left2:
        if "Credit_Score" in df.columns:
            fig = px.histogram(
                df,
                x="Credit_Score",
                nbins=30,
                title="Credit Score Distribution",
                color_discrete_sequence=[FINTECH_COLORS["light_blue"]],
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Credit score column not found in the current dataset.")

    with right2:
        if "Employment_Status" in df.columns:
            emp = _distribution_frame(df["Employment_Status"], "employment")
            fig = px.bar(
                emp,
                x="employment",
                y="count",
                title="Employment Status Distribution",
                color_discrete_sequence=[FINTECH_COLORS["teal"]],
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Employment status column not found in the current dataset.")

    left3, right3 = st.columns(2)
    with left3:
        if "Credit_Utilization_Percentage" in df.columns:
            fig = px.histogram(
                df,
                x="Credit_Utilization_Percentage",
                nbins=30,
                title="Credit Utilization Distribution (%)",
                color_discrete_sequence=["#93C5FD"],
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Credit utilization column not found in the current dataset.")

    with right3:
        if bundle is not None and not bundle.summary_df.empty:
            test_df = comparison_frame(bundle, split="test")
            metric = st.session_state.get("selected_metric", "macro_f1")
            if metric not in test_df.columns:
                metric = "macro_f1"
            st.plotly_chart(plot_metric_comparison(test_df, metric=metric), width="stretch")
        else:
            st.info("Train models in Model Laboratory to unlock the model comparison chart.")