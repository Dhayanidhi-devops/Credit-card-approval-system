"""Model training, comparison, persistence, and optional GridSearchCV."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from modules.config import MODELS_DIR, MODEL_DISPLAY_NAMES, RANDOM_SEED, TEST_SIZE
from modules.data_loader import DatasetProfile, get_feature_frame
from modules.evaluation import compute_classification_metrics, metrics_to_row, select_best_model
from modules.preprocessing import build_preprocessor, normalize_target_labels


ProgressCallback = Callable[[str], None]


@dataclass
class TrainingBundle:
    """Container for all trained models and metadata."""

    models: dict[str, Pipeline] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_model_name: str | None = None
    best_metric_name: str = "macro_f1"
    best_metric_value: float | None = None
    feature_columns: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    target_column: str | None = None
    class_labels: list[str] = field(default_factory=list)
    trained_at: str | None = None
    dataset_info: dict[str, Any] = field(default_factory=dict)
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None
    X_train: pd.DataFrame | None = None
    y_train: pd.Series | None = None


def get_estimator_catalog(random_state: int = RANDOM_SEED) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def get_param_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "logistic_regression": {
            "model__C": [0.1, 1.0, 3.0],
            "model__solver": ["lbfgs"],
        },
        "random_forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_leaf": [1, 2],
        },
        "extra_trees": {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 16],
            "model__min_samples_leaf": [1, 2],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 150],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    }


def _positive_proba(pipeline: Pipeline, x: pd.DataFrame, positive_label: str = "Approved") -> np.ndarray | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        proba = pipeline.predict_proba(x)
        classes = list(pipeline.named_steps["model"].classes_)
        if positive_label in classes:
            idx = classes.index(positive_label)
        else:
            idx = len(classes) - 1
        return proba[:, idx]
    except Exception:
        return None


def _full_proba(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        return pipeline.predict_proba(x)
    except Exception:
        return None


def train_single_model(
    model_key: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    tune: bool = False,
    cv_folds: int = 3,
    scoring: str = "f1_macro",
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    estimators = get_estimator_catalog()
    if model_key not in estimators:
        raise KeyError(f"Unknown model key: {model_key}")

    display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    if progress:
        progress(f"Training {display}...")

    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    base_model = estimators[model_key]
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    best_params: dict[str, Any] | None = None
    if tune:
        if progress:
            progress(f"Running GridSearchCV for {display}...")
        grid = get_param_grids().get(model_key, {})
        if grid:
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring=scoring,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED),
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_train, y_train)
            pipe = search.best_estimator_
            best_params = search.best_params_
        else:
            pipe.fit(x_train, y_train)
    else:
        pipe.fit(x_train, y_train)

    y_train_pred = pipe.predict(x_train)
    y_test_pred = pipe.predict(x_test)
    train_proba = _full_proba(pipe, x_train)
    test_proba = _full_proba(pipe, x_test)

    # Align ROC with Approved probability when available.
    train_pos = _positive_proba(pipe, x_train)
    test_pos = _positive_proba(pipe, x_test)

    train_metrics = compute_classification_metrics(
        y_train,
        y_train_pred,
        y_proba=np.column_stack([1 - train_pos, train_pos]) if train_pos is not None else train_proba,
    )
    test_metrics = compute_classification_metrics(
        y_test,
        y_test_pred,
        y_proba=np.column_stack([1 - test_pos, test_pos]) if test_pos is not None else test_proba,
    )

    return {
        "model_key": model_key,
        "display_name": display,
        "pipeline": pipe,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_test_true": np.asarray(y_test).astype(str),
        "y_test_pred": np.asarray(y_test_pred).astype(str),
        "y_test_proba_positive": test_pos,
        "best_params": best_params,
    }


def train_models(
    df: pd.DataFrame,
    profile: DatasetProfile,
    model_keys: list[str] | None = None,
    metric_for_best: str = "macro_f1",
    tune: bool = False,
    run_cv: bool = False,
    cv_folds: int = 5,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    progress: ProgressCallback | None = None,
) -> TrainingBundle:
    x, y_raw = get_feature_frame(df, profile)
    y = normalize_target_labels(y_raw)

    numeric_features = [c for c in profile.numeric_columns if c in x.columns]
    categorical_features = [c for c in profile.categorical_columns if c in x.columns]

    # Re-infer if profile lists drifted.
    if not numeric_features and not categorical_features:
        numeric_features = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
        categorical_features = [c for c in x.columns if c not in numeric_features]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    keys = model_keys or list(get_estimator_catalog().keys())
    bundle = TrainingBundle(
        feature_columns=list(x.columns),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=profile.target_column,
        class_labels=sorted(y.unique().tolist()),
        trained_at=datetime.now(timezone.utc).isoformat(),
        dataset_info={
            "source_name": profile.source_name,
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "target_distribution": profile.target_distribution,
        },
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        best_metric_name=metric_for_best,
    )

    rows: list[dict[str, Any]] = []
    for key in keys:
        try:
            result = train_single_model(
                model_key=key,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                tune=tune,
                cv_folds=min(cv_folds, 3) if tune else cv_folds,
                progress=progress,
            )
        except Exception as exc:  # Keep other models training if one fails.
            if progress:
                progress(f"Failed training {key}: {exc}")
            continue

        bundle.models[key] = result["pipeline"]
        bundle.results[key] = result
        rows.append(metrics_to_row(result["display_name"], "train", result["train_metrics"]))
        rows.append(metrics_to_row(result["display_name"], "test", result["test_metrics"]))

        if run_cv:
            if progress:
                progress(f"Cross-validating {result['display_name']}...")
            try:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scores = cross_val_score(
                    result["pipeline"],
                    x_train,
                    y_train,
                    cv=cv,
                    scoring="f1_macro",
                    n_jobs=-1,
                )
                bundle.cv_scores[key] = [float(s) for s in scores]
            except Exception as exc:
                if progress:
                    progress(f"CV failed for {key}: {exc}")

    bundle.summary_df = pd.DataFrame(rows)
    best_name, best_score = select_best_model(bundle.results, metric=metric_for_best)
    bundle.best_model_name = best_name
    bundle.best_metric_value = best_score
    return bundle


def save_model_artifact(
    bundle: TrainingBundle,
    model_key: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Persist a selected model pipeline and metadata with joblib + JSON sidecar."""
    if not bundle.models:
        raise ValueError("No trained models available to save.")

    key = model_key or bundle.best_model_name
    if key is None or key not in bundle.models:
        raise ValueError("Selected model is not available in the training bundle.")

    out_dir = Path(output_dir) if output_dir else MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"creditwise_{key}_{stamp}.joblib"
    meta_path = out_dir / f"creditwise_{key}_{stamp}.json"

    payload = {
        "pipeline": bundle.models[key],
        "model_key": key,
        "display_name": MODEL_DISPLAY_NAMES.get(key, key),
        "feature_columns": bundle.feature_columns,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "target_column": bundle.target_column,
        "class_labels": bundle.class_labels,
        "trained_at": bundle.trained_at,
        "metrics": bundle.results[key]["test_metrics"],
        "dataset_info": bundle.dataset_info,
        "best_params": bundle.results[key].get("best_params"),
    }
    joblib.dump(payload, model_path)

    serializable_metrics = {
        k: v
        for k, v in bundle.results[key]["test_metrics"].items()
        if k not in {"confusion_matrix", "classification_report", "classification_report_text", "labels"}
    }
    meta = {
        "model_path": str(model_path.name),
        "model_key": key,
        "display_name": MODEL_DISPLAY_NAMES.get(key, key),
        "trained_at": bundle.trained_at,
        "feature_columns": bundle.feature_columns,
        "metrics": serializable_metrics,
        "dataset_info": bundle.dataset_info,
        "disclaimer": "Academic demonstration model artifact. Not for production lending.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path


def load_model_artifact(path: str | Path, expected_features: list[str] | None = None) -> dict[str, Any]:
    """Load a previously saved model and validate basic compatibility."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix.lower() not in {".joblib", ".pkl"}:
        raise ValueError("Only .joblib / .pkl model artifacts are supported.")

    # Never auto-load untrusted files; caller must explicitly request load.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Invalid model artifact format. Expected a CreditWise Studio joblib payload.")

    feature_columns = artifact.get("feature_columns") or []
    if expected_features is not None:
        missing = [c for c in feature_columns if c not in expected_features]
        extra_note = ""
        if missing:
            raise ValueError(
                "Loaded model is incompatible with the current dataset. Missing features: "
                + ", ".join(missing)
            )
        # Soft check for renamed schemas.
        if set(feature_columns) != set(expected_features):
            extra_note = " Feature sets differ; prediction may fail for mismatched columns."
            artifact["compatibility_warning"] = extra_note.strip()

    artifact["source_path"] = str(path)
    return artifact


def list_saved_models(directory: str | Path | None = None) -> list[Path]:
    directory = Path(directory) if directory else MODELS_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("creditwise_*.joblib"), reverse=True)


def comparison_frame(bundle: TrainingBundle, split: str = "test") -> pd.DataFrame:
    if bundle.summary_df.empty:
        return pd.DataFrame()
    return bundle.summary_df[bundle.summary_df["split"] == split].reset_index(drop=True)