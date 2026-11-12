"""Model Laboratory — train, compare, tune, and persist models."""

from __future__ import annotations

import streamlit as st

from modules.config import METRIC_OPTIONS, MODEL_DISPLAY_NAMES
from modules.data_loader import validate_for_training
from modules.evaluation import (
    class_imbalance_ratio,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from modules.model_training import (
    comparison_frame,
    list_saved_models,
    load_model_artifact,
    save_model_artifact,
    train_models,
)
from modules.report_generator import metrics_dataframe_to_csv_bytes
from modules.ui import disclaimer_banner, require_dataset, section_title


def render() -> None:
    section_title(
        "Model Laboratory",
        "Train supervised classifiers with leakage-safe sklearn Pipelines and compare evaluation metrics.",
    )
    disclaimer_banner()

    if not require_dataset():
        return

    df = st.session_state.dataset
    profile = st.session_state.dataset_profile

    try:
        validate_for_training(df, profile)
    except Exception as exc:
        st.error(str(exc))
        return

    imbalance = class_imbalance_ratio(profile.target_distribution)
    default_metric = "macro_f1" if imbalance >= 1.5 else st.session_state.get("selected_metric", "macro_f1")
    if imbalance >= 1.5:
        st.warning(
            f"Class imbalance ratio ≈ {imbalance:.2f}. Prefer Macro F1 or Balanced Accuracy over Accuracy alone."
        )

    st.markdown("#### Training Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_models = st.multiselect(
            "Models to train",
            options=list(MODEL_DISPLAY_NAMES.keys()),
            default=list(MODEL_DISPLAY_NAMES.keys()),
            format_func=lambda k: MODEL_DISPLAY_NAMES[k],
        )
    with c2:
        metric = st.selectbox(
            "Preferred evaluation metric",
            options=list(METRIC_OPTIONS.keys()),
            index=list(METRIC_OPTIONS.keys()).index(default_metric)
            if default_metric in METRIC_OPTIONS
            else 0,
            format_func=lambda k: METRIC_OPTIONS[k],
        )
        st.session_state.selected_metric = metric
    with c3:
        tune = st.checkbox("Enable GridSearchCV hyperparameter tuning", value=False)
        run_cv = st.checkbox("Run stratified cross-validation on train set", value=False)

    progress_box = st.empty()

    def _progress(message: str) -> None:
        progress_box.info(message)

    col_a, col_b = st.columns(2)
    train_all = col_a.button("Train Selected / All Models", type="primary")
    clear_models = col_b.button("Clear Trained Models")

    if clear_models:
        st.session_state.training_bundle = None
        st.session_state.active_model_key = None
        st.success("Cleared in-session trained models.")

    if train_all:
        if not selected_models:
            st.error("Select at least one model to train.")
        else:
            with st.spinner("Training models..."):
                try:
                    bundle = train_models(
                        df=df,
                        profile=profile,
                        model_keys=selected_models,
                        metric_for_best=metric,
                        tune=tune,
                        run_cv=run_cv,
                        progress=_progress,
                    )
                    st.session_state.training_bundle = bundle
                    st.session_state.active_model_key = bundle.best_model_name
                    progress_box.success(
                        f"Training complete. Best model by {METRIC_OPTIONS[metric]}: "
                        f"{MODEL_DISPLAY_NAMES.get(bundle.best_model_name, bundle.best_model_name)}"
                    )
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    bundle = st.session_state.get("training_bundle")
    if bundle is None or not bundle.results:
        st.info("Train one or more models to view laboratory results.")
    else:
        st.markdown("#### Comparison Summary")
        test_df = comparison_frame(bundle, "test")
        train_df = comparison_frame(bundle, "train")
        st.dataframe(test_df, width="stretch")
        st.plotly_chart(plot_metric_comparison(test_df, metric=metric if metric in test_df.columns else "macro_f1"), width="stretch")

        st.download_button(
            "Download performance CSV",
            data=metrics_dataframe_to_csv_bytes(bundle.summary_df),
            file_name="creditwise_model_performance.csv",
            mime="text/csv",
        )

        st.markdown("#### Detailed Model Diagnostics")
        focus = st.selectbox(
            "Inspect model",
            options=list(bundle.results.keys()),
            format_func=lambda k: bundle.results[k]["display_name"],
        )
        result = bundle.results[focus]
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Train metrics**")
            st.json(
                {
                    k: result["train_metrics"].get(k)
                    for k in [
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "balanced_accuracy",
                        "precision_macro",
                        "recall_macro",
                        "roc_auc",
                    ]
                }
            )
        with m2:
            st.markdown("**Test metrics**")
            st.json(
                {
                    k: result["test_metrics"].get(k)
                    for k in [
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "balanced_accuracy",
                        "precision_macro",
                        "recall_macro",
                        "roc_auc",
                    ]
                }
            )

        st.markdown("**Classification report (test)**")
        st.code(result["test_metrics"].get("classification_report_text", ""), language="text")

        labels = result["test_metrics"].get("labels") or sorted(set(result["y_test_true"]))
        cm = result["test_metrics"].get("confusion_matrix")
        if cm:
            st.plotly_chart(
                plot_confusion_matrix(cm, labels, f"Confusion Matrix — {result['display_name']}"),
                width="stretch",
            )

        proba = result.get("y_test_proba_positive")
        if proba is not None:
            roc = plot_roc_curve(result["y_test_true"], proba, title=f"ROC — {result['display_name']}")
            pr = plot_precision_recall_curve(
                result["y_test_true"], proba, title=f"Precision-Recall — {result['display_name']}"
            )
            c1, c2 = st.columns(2)
            if roc is not None:
                c1.plotly_chart(roc, width="stretch")
            if pr is not None:
                c2.plotly_chart(pr, width="stretch")

        if result.get("best_params"):
            st.markdown("**Best GridSearchCV parameters**")
            st.json(result["best_params"])

        if bundle.cv_scores.get(focus):
            st.markdown("**Cross-validation macro F1 scores (train folds)**")
            st.write(bundle.cv_scores[focus])

        st.markdown("#### Compare Train vs Test")
        merged = train_df.merge(test_df, on="model", suffixes=("_train", "_test"))
        st.dataframe(merged, width="stretch")

        st.markdown("#### Model Persistence")
        save_key = st.selectbox(
            "Model to save",
            options=list(bundle.models.keys()),
            format_func=lambda k: MODEL_DISPLAY_NAMES.get(k, k),
            key="save_model_key",
        )
        if st.button("Save selected model (joblib)"):
            try:
                path = save_model_artifact(bundle, model_key=save_key)
                st.success(f"Saved model artifact: `{path.name}`")
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    st.markdown("#### Load Previously Trained Model")
    saved = list_saved_models()
    if not saved:
        st.caption("No saved CreditWise model artifacts found in the models/ directory.")
    else:
        choice = st.selectbox("Saved artifacts", options=saved, format_func=lambda p: p.name)
        st.warning("Load models only from trusted local artifacts that you created in this project.")
        if st.button("Load selected model"):
            try:
                expected = None
                if st.session_state.get("training_bundle") is not None:
                    expected = st.session_state.training_bundle.feature_columns
                elif profile is not None:
                    expected = profile.numeric_columns + profile.categorical_columns
                artifact = load_model_artifact(choice, expected_features=expected)
                st.session_state.loaded_artifact = artifact
                st.success(f"Loaded `{choice.name}` for predictions.")
                if artifact.get("compatibility_warning"):
                    st.warning(artifact["compatibility_warning"])
            except Exception as exc:
                st.error(f"Could not load model: {exc}")