"""
CreditWise Studio — Intelligent Credit Card Approval Prediction System

Main Streamlit entrypoint with custom FinTech sidebar navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable when launched from any working directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.config import DEFAULT_DATASET_PATH, DISCLAIMER
from modules.data_loader import load_default_dataset, profile_dataset
from modules.ui import ensure_session_defaults, inject_styles, render_brand_header
from views import (
    about,
    applicant_prediction,
    dashboard,
    dataset_explorer,
    model_laboratory,
    model_performance,
    prediction_history,
)

st.set_page_config(
    page_title="CreditWise Studio",
    page_icon="CW",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_session_defaults()
inject_styles()


def bootstrap_default_dataset() -> None:
    """Load the bundled synthetic dataset once per session if available."""
    if st.session_state.dataset is not None:
        return
    if not DEFAULT_DATASET_PATH.exists():
        st.session_state.dataset_load_error = (
            f"Default dataset not found at {DEFAULT_DATASET_PATH}. "
            "Upload a dataset from Dataset Explorer."
        )
        return
    try:
        df = load_default_dataset()
        profile = profile_dataset(df, source_name=DEFAULT_DATASET_PATH.name)
        st.session_state.dataset = df
        st.session_state.dataset_profile = profile
        st.session_state.dataset_source = DEFAULT_DATASET_PATH.name
        st.session_state.dataset_load_error = None
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.session_state.dataset_load_error = str(exc)


bootstrap_default_dataset()

PAGES = {
    "Dashboard": dashboard.render,
    "Applicant Prediction": applicant_prediction.render,
    "Model Laboratory": model_laboratory.render,
    "Dataset Explorer": dataset_explorer.render,
    "Model Performance": model_performance.render,
    "Prediction History": prediction_history.render,
    "About & Methodology": about.render,
}

with st.sidebar:
    render_brand_header()
    st.markdown("---")
    page = st.radio(
        "Navigation",
        options=list(PAGES.keys()),
        index=list(PAGES.keys()).index(st.session_state.selected_page)
        if st.session_state.selected_page in PAGES
        else 0,
        label_visibility="collapsed",
    )
    st.session_state.selected_page = page

    st.markdown("---")
    st.caption("Dataset status")
    if st.session_state.dataset is not None:
        profile = st.session_state.dataset_profile
        st.success(f"Loaded: {st.session_state.dataset_source}")
        st.caption(f"{profile.n_rows} rows · {profile.n_cols} columns")
        if profile.target_column:
            st.caption(f"Target: {profile.target_column}")
    else:
        st.warning("No dataset loaded")
        if st.session_state.get("dataset_load_error"):
            st.error(st.session_state.dataset_load_error)

    bundle = st.session_state.get("training_bundle")
    if bundle and bundle.best_model_name:
        st.caption(
            f"Best model: {bundle.results[bundle.best_model_name]['display_name']}"
        )

    st.markdown("---")
    st.caption(DISCLAIMER)

render_brand_header(subtitle="B.Tech AIML Academic Demonstration · Synthetic Data")
PAGES[page]()

st.markdown(
    '<div class="cw-footer-note">CreditWise Studio · Predictions are academic model estimates only.</div>',
    unsafe_allow_html=True,
)