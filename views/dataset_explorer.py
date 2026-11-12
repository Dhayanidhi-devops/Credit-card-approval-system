"""Dataset Explorer — profiling, validation, and downloadable copies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.config import DEFAULT_DATASET_PATH
from modules.data_loader import (
    DatasetValidationError,
    load_dataframe_from_upload,
    load_default_dataset,
    profile_dataset,
)
from modules.preprocessing import normalize_target_labels
from modules.report_generator import cleaned_dataset_csv_bytes
from modules.ui import disclaimer_banner, section_title


def _set_dataset(df: pd.DataFrame, source_name: str) -> None:
    profile = profile_dataset(df, source_name=source_name)
    st.session_state.dataset = df
    st.session_state.dataset_profile = profile
    st.session_state.dataset_source = source_name
    # New data invalidates previous training unless user reloads a compatible model.
    st.session_state.training_bundle = None


def render() -> None:
    section_title(
        "Dataset Explorer",
        "Inspect schema quality, distributions, and correlations. Uploaded data is never silently altered on disk.",
    )
    disclaimer_banner()

    st.markdown("#### Load Data")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load default synthetic dataset"):
            try:
                df = load_default_dataset()
                _set_dataset(df, source_name=DEFAULT_DATASET_PATH.name)
                st.success(f"Loaded default dataset: {DEFAULT_DATASET_PATH.name}")
            except Exception as exc:
                st.error(f"Failed to load default dataset: {exc}")
    with c2:
        uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
        if uploaded is not None and st.button("Use uploaded file"):
            try:
                df = load_dataframe_from_upload(uploaded, uploaded.name)
                _set_dataset(df, source_name=uploaded.name)
                st.success(f"Loaded uploaded file: {uploaded.name}")
            except DatasetValidationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    if st.session_state.get("dataset") is None:
        st.info("Load a dataset to explore statistics.")
        return

    df = st.session_state.dataset
    profile = st.session_state.dataset_profile

    st.markdown("#### Dataset Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{profile.n_rows:,}")
    m2.metric("Columns", f"{profile.n_cols:,}")
    m3.metric("Missing cells", f"{sum(profile.missing_counts.values()):,}")
    m4.metric("Duplicate rows", f"{profile.duplicate_count:,}")

    st.write(f"**Source:** {profile.source_name}")
    st.write(f"**Target column:** {profile.target_column or 'Not detected'}")
    st.write(f"**ID column:** {profile.id_column or 'Not detected'}")
    st.write(f"**Numeric features:** {', '.join(profile.numeric_columns) or 'None'}")
    st.write(f"**Categorical features:** {', '.join(profile.categorical_columns) or 'None'}")

    for warning in profile.warnings:
        st.warning(warning)

    st.markdown("#### Preview")
    st.dataframe(df.head(50), width="stretch")

    st.markdown("#### Data Types")
    dtype_df = pd.DataFrame(
        {"column": list(profile.dtypes.keys()), "dtype": list(profile.dtypes.values())}
    )
    st.dataframe(dtype_df, width="stretch")

    st.markdown("#### Missing Value Report")
    if profile.missing_counts:
        miss_df = pd.DataFrame(
            {
                "column": list(profile.missing_counts.keys()),
                "missing_count": list(profile.missing_counts.values()),
            }
        )
        miss_df["missing_pct"] = (miss_df["missing_count"] / profile.n_rows * 100).round(2)
        st.dataframe(miss_df, width="stretch")
    else:
        st.success("No missing values detected.")

    st.markdown("#### Numerical Statistics")
    numeric_cols = profile.numeric_columns
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T, width="stretch")
    else:
        st.info("No numeric columns available.")

    st.markdown("#### Categorical Distributions")
    cat_cols = profile.categorical_columns
    if cat_cols:
        selected_cat = st.selectbox("Categorical column", options=cat_cols)
        counts = df[selected_cat].astype(str).value_counts(dropna=False).reset_index()
        counts.columns = [selected_cat, "count"]
        fig = px.bar(counts, x=selected_cat, y="count", title=f"Distribution — {selected_cat}")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No categorical columns available.")

    if profile.target_column:
        st.markdown("#### Target Distribution")
        target = normalize_target_labels(df[profile.target_column])
        tcounts = target.value_counts().reset_index()
        tcounts.columns = ["target", "count"]
        fig = px.pie(tcounts, names="target", values="count", title="Target Distribution")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Correlation Heatmap")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Numeric Feature Correlation",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Need at least two numeric columns for a correlation heatmap.")

    st.markdown("#### Export In-Memory Copy")
    st.caption(
        "Downloads a CSV copy of the currently loaded dataframe. "
        "The original Excel/CSV file on disk is not overwritten."
    )
    # Optional light cleaning for export only.
    export_df = df.copy()
    if st.checkbox("Fill missing values in export copy only (median / mode)", value=False):
        for col in export_df.columns:
            if export_df[col].isna().any():
                if pd.api.types.is_numeric_dtype(export_df[col]):
                    export_df[col] = export_df[col].fillna(export_df[col].median())
                else:
                    mode = export_df[col].mode(dropna=True)
                    if not mode.empty:
                        export_df[col] = export_df[col].fillna(mode.iloc[0])
    st.download_button(
        "Download dataset CSV copy",
        data=cleaned_dataset_csv_bytes(export_df),
        file_name="creditwise_dataset_copy.csv",
        mime="text/csv",
    )