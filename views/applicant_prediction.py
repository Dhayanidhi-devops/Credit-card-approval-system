"""Applicant registration and prediction page."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from modules.config import CATEGORICAL_FEATURE_HINTS, NUMERIC_FEATURE_HINTS
from modules.prediction import (
    PredictionValidationError,
    make_history_record,
    predict_applicant,
    validate_applicant_input,
)
from modules.preprocessing import safe_numeric_bounds
from modules.report_generator import prediction_to_csv_bytes, prediction_to_pdf_bytes
from modules.ui import (
    available_model_options,
    disclaimer_banner,
    require_dataset,
    require_trained_models,
    resolve_active_pipeline,
    section_title,
)


def _default_value(series: pd.Series, fallback):
    clean = series.dropna()
    if clean.empty:
        return fallback
    if pd.api.types.is_numeric_dtype(clean):
        return float(clean.median())
    return str(clean.mode().iloc[0])


def render() -> None:
    section_title(
        "Applicant Prediction",
        "Submit applicant attributes to a trained pipeline. Models are not retrained for each prediction.",
    )
    disclaimer_banner()

    if not require_dataset() or not require_trained_models():
        return

    df = st.session_state.dataset
    bundle = st.session_state.get("training_bundle")
    artifact = st.session_state.get("loaded_artifact")

    options = available_model_options()
    if not options:
        st.warning("No models available.")
        return

    model_key = st.selectbox(
        "Selected model",
        options=list(options.keys()),
        format_func=lambda k: options[k],
    )

    # Determine feature schema from active source.
    if model_key.startswith("loaded::") and artifact is not None:
        feature_columns = list(artifact.get("feature_columns") or [])
        numeric_features = list(artifact.get("numeric_features") or [])
        categorical_features = list(artifact.get("categorical_features") or [])
    else:
        feature_columns = list(bundle.feature_columns)
        numeric_features = list(bundle.numeric_features)
        categorical_features = list(bundle.categorical_features)

    st.markdown("#### Applicant Information")
    values: dict = {}

    # Preferred form order from product requirements.
    preferred_order = list(NUMERIC_FEATURE_HINTS.keys()) + list(CATEGORICAL_FEATURE_HINTS.keys())
    ordered_features = [c for c in preferred_order if c in feature_columns]
    ordered_features += [c for c in feature_columns if c not in ordered_features]

    cols = st.columns(2)
    for idx, col_name in enumerate(ordered_features):
        container = cols[idx % 2]
        with container:
            if col_name in numeric_features or col_name in NUMERIC_FEATURE_HINTS:
                hint = NUMERIC_FEATURE_HINTS.get(col_name, {})
                label = hint.get("label", col_name)
                unit = hint.get("unit", "")
                if col_name in df.columns:
                    low, high = safe_numeric_bounds(
                        df[col_name],
                        hint.get("min", 0),
                        hint.get("max", 100),
                    )
                    default = _default_value(df[col_name], hint.get("min", 0))
                else:
                    low = float(hint.get("min", 0))
                    high = float(hint.get("max", 100))
                    default = low
                # Expand a little beyond observed range for form usability.
                form_min = min(float(hint.get("min", low)), low)
                form_max = max(float(hint.get("max", high)), high)
                step = float(hint.get("step", 1))
                values[col_name] = st.number_input(
                    f"{label}" + (f" ({unit})" if unit and unit not in label else ""),
                    min_value=form_min,
                    max_value=form_max,
                    value=float(default),
                    step=step,
                    help=f"Expected range roughly {low:.0f} – {high:.0f} based on the loaded dataset.",
                    key=f"pred_num_{col_name}",
                )
            else:
                hint = CATEGORICAL_FEATURE_HINTS.get(col_name, {})
                label = hint.get("label", col_name)
                if col_name in df.columns:
                    options_cat = sorted(df[col_name].dropna().astype(str).unique().tolist())
                    default = _default_value(df[col_name], options_cat[0] if options_cat else "")
                else:
                    options_cat = hint.get("options", ["Unknown"])
                    default = options_cat[0]
                if not options_cat:
                    options_cat = ["Unknown"]
                index = options_cat.index(str(default)) if str(default) in options_cat else 0
                values[col_name] = st.selectbox(
                    label,
                    options=options_cat,
                    index=index,
                    key=f"pred_cat_{col_name}",
                )

    predict_clicked = st.button("Predict Approval Outcome", type="primary")

    if predict_clicked:
        try:
            applicant_df = validate_applicant_input(
                values=values,
                feature_columns=feature_columns,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            pipeline, display_name, _ = resolve_active_pipeline(model_key)
            result = predict_applicant(pipeline, applicant_df, model_name=display_name)
            reference = f"REF-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            history_row = make_history_record(result, reference=reference)
            st.session_state.prediction_history.insert(0, history_row)
            st.session_state.last_prediction = {
                "reference": reference,
                "result": result,
            }
            st.success(f"Prediction complete for {reference}.")
        except PredictionValidationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to generate prediction: {exc}")

    last = st.session_state.get("last_prediction")
    if last:
        result = last["result"]
        reference = last["reference"]
        st.markdown("---")
        st.markdown("#### Prediction Report")
        badge = "cw-badge-approved" if result["prediction"] == "Approved" else "cw-badge-declined"
        st.markdown(
            f'<span class="cw-badge {badge}">{result["prediction"].upper()}</span>',
            unsafe_allow_html=True,
        )
        st.write(f"**Reference:** {reference}")
        st.write(f"**Model:** {result['model_name']}")
        st.write(f"**Timestamp:** {result['timestamp']}")
        if result.get("probabilities"):
            st.write("**Class probabilities (model estimates only, not guaranteed approval odds):**")
            st.json({k: round(v, 4) for k, v in result["probabilities"].items()})
        st.info(result["explanation"].replace("**", ""))

        csv_bytes = prediction_to_csv_bytes(result, reference)
        st.download_button(
            "Download Prediction CSV",
            data=csv_bytes,
            file_name=f"{reference}_prediction.csv",
            mime="text/csv",
        )
        try:
            pdf_bytes = prediction_to_pdf_bytes(result, reference)
            st.download_button(
                "Download Prediction PDF",
                data=pdf_bytes,
                file_name=f"{reference}_prediction.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.caption(f"PDF export unavailable: {exc}")