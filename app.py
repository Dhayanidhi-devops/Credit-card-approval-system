"""
CreditWise Studio — Intelligent Credit Card Approval Prediction System

Main Streamlit entrypoint with branded sidebar navigation via st.navigation.
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

# File-based pages keep unique URL paths and work cleanly with Streamlit AppTest.
nav = st.navigation(
    {
        "CreditWise Studio": [
            st.Page(
                "page_scripts/dashboard.py",
                title="Dashboard",
                icon=":material/dashboard:",
                default=True,
                url_path="dashboard",
            ),
            st.Page(
                "page_scripts/applicant_prediction.py",
                title="Applicant Prediction",
                icon=":material/person_search:",
                url_path="applicant-prediction",
            ),
            st.Page(
                "page_scripts/model_laboratory.py",
                title="Model Laboratory",
                icon=":material/science:",
                url_path="model-laboratory",
            ),
            st.Page(
                "page_scripts/dataset_explorer.py",
                title="Dataset Explorer",
                icon=":material/table_view:",
                url_path="dataset-explorer",
            ),
            st.Page(
                "page_scripts/model_performance.py",
                title="Model Performance",
                icon=":material/monitoring:",
                url_path="model-performance",
            ),
            st.Page(
                "page_scripts/prediction_history.py",
                title="Prediction History",
                icon=":material/history:",
                url_path="prediction-history",
            ),
            st.Page(
                "page_scripts/about.py",
                title="About & Methodology",
                icon=":material/info:",
                url_path="about",
            ),
        ]
    }
)

with st.sidebar:
    render_brand_header()
    st.markdown("---")
    st.caption("DATASET STATUS")
    if st.session_state.dataset is not None:
        profile = st.session_state.dataset_profile
        st.markdown(
            f'<div class="cw-status-ok">Loaded: {st.session_state.dataset_source}<br>'
            f"{profile.n_rows} rows · {profile.n_cols} columns"
            + (f"<br>Target: {profile.target_column}" if profile.target_column else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="cw-status-warn">No dataset loaded</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("dataset_load_error"):
            st.error(st.session_state.dataset_load_error)

    bundle = st.session_state.get("training_bundle")
    if bundle and bundle.best_model_name:
        st.caption(
            f"Best model: {bundle.results[bundle.best_model_name]['display_name']}"
        )

    st.markdown("---")
    st.caption("Academic synthetic-data demonstration. Not for real lending decisions.")

render_brand_header(subtitle="B.Tech AIML Academic Demonstration · Synthetic Data")
nav.run()

st.markdown(
    '<div class="cw-footer-note">CreditWise Studio · Predictions are academic model estimates only.</div>',
    unsafe_allow_html=True,
)
