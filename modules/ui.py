"""Shared Streamlit UI helpers for CreditWise Studio."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from modules.config import ASSETS_DIR, DISCLAIMER


def inject_styles() -> None:
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_brand_header(subtitle: str | None = None) -> None:
    st.markdown(
        """
        <div class="cw-brand">
            <div class="cw-logo">CW</div>
            <div>
                <div class="cw-brand-title">CreditWise Studio</div>
                <div class="cw-brand-subtitle">Intelligent Credit Card Approval Prediction</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


def metric_card(label: str, value: str, hint: str | None = None) -> None:
    hint_html = f'<div class="cw-card-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="cw-card">
            <div class="cw-card-label">{label}</div>
            <div class="cw-card-value">{value}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer_banner() -> None:
    st.info(DISCLAIMER)


def section_title(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def ensure_session_defaults() -> None:
    defaults = {
        "dataset": None,
        "dataset_profile": None,
        "dataset_source": None,
        "training_bundle": None,
        "loaded_artifact": None,
        "prediction_history": [],
        "last_prediction": None,
        "selected_page": "Dashboard",
        "selected_metric": "macro_f1",
        "active_model_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def require_dataset() -> bool:
    if st.session_state.get("dataset") is None:
        st.warning("No dataset loaded. Open **Dataset Explorer** or reload the default dataset from the sidebar.")
        return False
    return True


def require_trained_models() -> bool:
    bundle = st.session_state.get("training_bundle")
    artifact = st.session_state.get("loaded_artifact")
    if bundle is None and artifact is None:
        st.warning("No trained models available. Train models in **Model Laboratory** first.")
        return False
    return True


def available_model_options() -> dict[str, str]:
    """Return mapping of model_key -> display name for currently usable models."""
    options: dict[str, str] = {}
    bundle = st.session_state.get("training_bundle")
    if bundle is not None:
        for key, result in bundle.results.items():
            options[key] = result.get("display_name", key)
    artifact = st.session_state.get("loaded_artifact")
    if artifact is not None:
        key = artifact.get("model_key", "loaded_model")
        options[f"loaded::{key}"] = f"Loaded: {artifact.get('display_name', key)}"
    return options


def resolve_active_pipeline(model_key: str):
    if model_key.startswith("loaded::"):
        artifact = st.session_state.get("loaded_artifact")
        if artifact is None:
            raise ValueError("Loaded model artifact is unavailable.")
        return artifact["pipeline"], artifact.get("display_name", "Loaded Model"), artifact
    bundle = st.session_state.get("training_bundle")
    if bundle is None or model_key not in bundle.models:
        raise ValueError("Selected model is not trained in the current session.")
    return bundle.models[model_key], bundle.results[model_key]["display_name"], None