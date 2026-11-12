"""Model Performance and explainability page."""

from __future__ import annotations

import streamlit as st

from modules.evaluation import plot_confusion_matrix, plot_metric_comparison
from modules.explainability import (
    SHAP_AVAILABLE,
    compute_permutation_importance,
    compute_shap_summary,
    compute_tree_feature_importance,
    plot_importance_bar,
)
from modules.model_training import comparison_frame
from modules.ui import disclaimer_banner, require_trained_models, section_title


def render() -> None:
    section_title(
        "Model Performance",
        "Compare hold-out metrics and inspect feature influence. Influence is association, not causation.",
    )
    disclaimer_banner()

    if not require_trained_models():
        return

    bundle = st.session_state.get("training_bundle")
    artifact = st.session_state.get("loaded_artifact")

    if bundle is not None and bundle.results:
        st.markdown("#### Hold-out Test Comparison")
        test_df = comparison_frame(bundle, "test")
        st.dataframe(test_df, width="stretch")
        metric = st.session_state.get("selected_metric", "macro_f1")
        if metric not in test_df.columns:
            metric = "macro_f1"
        st.plotly_chart(plot_metric_comparison(test_df, metric=metric), width="stretch")

        model_key = st.selectbox(
            "Model for detailed analysis",
            options=list(bundle.results.keys()),
            format_func=lambda k: bundle.results[k]["display_name"],
        )
        result = bundle.results[model_key]
        st.markdown("**Classification report**")
        st.code(result["test_metrics"].get("classification_report_text", ""), language="text")

        labels = result["test_metrics"].get("labels") or []
        cm = result["test_metrics"].get("confusion_matrix")
        if cm and labels:
            st.plotly_chart(
                plot_confusion_matrix(cm, labels, f"Confusion Matrix — {result['display_name']}"),
                width="stretch",
            )

        st.markdown("#### Explainability")
        st.caption(
            "Feature importance rankings describe how strongly features associate with model "
            "predictions on this synthetic dataset. They do not prove that a feature caused an approval or decline."
        )

        pipeline = bundle.models[model_key]
        x_test = bundle.X_test
        y_test = bundle.y_test

        if st.button("Compute permutation feature importance"):
            with st.spinner("Calculating permutation importance..."):
                try:
                    imp = compute_permutation_importance(pipeline, x_test, y_test)
                    st.session_state[f"perm_imp_{model_key}"] = imp
                except Exception as exc:
                    st.error(f"Permutation importance failed: {exc}")

        imp = st.session_state.get(f"perm_imp_{model_key}")
        if imp is not None:
            st.dataframe(imp.head(20), width="stretch")
            st.plotly_chart(
                plot_importance_bar(
                    imp,
                    value_col="importance_mean",
                    title="Global Permutation Importance",
                ),
                width="stretch",
            )
            st.write("Most influential features (by permutation importance):")
            st.write(", ".join(imp.head(5)["feature"].tolist()))

        tree_imp = compute_tree_feature_importance(pipeline)
        if tree_imp is not None:
            st.markdown("**Tree-based impurity importance (transformed features)**")
            st.dataframe(tree_imp.head(20), width="stretch")
            st.plotly_chart(
                plot_importance_bar(
                    tree_imp,
                    value_col="importance",
                    title="Tree Feature Importance",
                ),
                width="stretch",
            )
        else:
            st.caption("Tree impurity importance is unavailable for the selected estimator.")

        st.markdown("**Optional SHAP module**")
        if not SHAP_AVAILABLE:
            st.info("SHAP is not available in this environment. The rest of the application continues to work.")
        else:
            if st.button("Compute SHAP summary (optional)"):
                with st.spinner("Computing SHAP values..."):
                    shap_result = compute_shap_summary(pipeline, x_test)
                    st.session_state[f"shap_{model_key}"] = shap_result
            shap_result = st.session_state.get(f"shap_{model_key}")
            if shap_result:
                st.caption(shap_result["note"])
                st.dataframe(shap_result["summary"].head(20), width="stretch")
                st.plotly_chart(
                    plot_importance_bar(
                        shap_result["summary"],
                        value_col="mean_abs_shap",
                        title="Mean |SHAP| Feature Influence",
                    ),
                    width="stretch",
                )
            elif shap_result is None and SHAP_AVAILABLE:
                st.caption("Click the button above to compute optional SHAP explanations.")

        if bundle.best_model_name:
            st.success(
                f"Current best model by {bundle.best_metric_name}: "
                f"{bundle.results[bundle.best_model_name]['display_name']} "
                f"({bundle.best_metric_value:.4f})"
            )
    elif artifact is not None:
        st.info(
            "A saved model is loaded, but in-session comparison metrics are unavailable. "
            "Retrain in Model Laboratory for full performance charts, or use Applicant Prediction."
        )
        st.json(
            {
                "model": artifact.get("display_name"),
                "trained_at": artifact.get("trained_at"),
                "metrics": {
                    k: v
                    for k, v in (artifact.get("metrics") or {}).items()
                    if k
                    not in {
                        "confusion_matrix",
                        "classification_report",
                        "classification_report_text",
                        "labels",
                    }
                },
            }
        )
    else:
        st.warning("No performance data available.")