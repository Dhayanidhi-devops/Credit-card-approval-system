"""Session prediction history page."""

from __future__ import annotations

import streamlit as st

from modules.prediction import history_to_dataframe
from modules.ui import disclaimer_banner, section_title


def render() -> None:
    section_title(
        "Prediction History",
        "Session-only prediction log. Sensitive personal identifiers are not stored beyond the applicant reference.",
    )
    disclaimer_banner()

    history = st.session_state.get("prediction_history") or []
    df = history_to_dataframe(history)

    if df.empty:
        st.info("No predictions yet. Generate one from the Applicant Prediction page.")
        return

    c1, c2 = st.columns(2)
    with c1:
        query = st.text_input("Search by reference or model", value="")
    with c2:
        outcomes = ["All"] + sorted(df["prediction"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Filter by prediction", options=outcomes)

    filtered = df.copy()
    if selected != "All":
        filtered = filtered[filtered["prediction"].astype(str) == selected]
    if query.strip():
        q = query.strip().lower()
        mask = (
            filtered["reference"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["model"].astype(str).str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]

    st.dataframe(filtered, width="stretch")
    st.caption(f"Showing {len(filtered)} of {len(df)} session records.")

    st.download_button(
        "Export history CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="creditwise_prediction_history.csv",
        mime="text/csv",
    )

    if st.button("Clear session history"):
        st.session_state.prediction_history = []
        st.session_state.last_prediction = None
        st.success("Prediction history cleared for this session.")
        st.rerun()