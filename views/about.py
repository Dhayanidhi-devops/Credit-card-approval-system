"""About & Methodology page."""

from __future__ import annotations

import streamlit as st

from modules.config import DISCLAIMER
from modules.ui import section_title


def render() -> None:
    section_title(
        "About & Methodology",
        "Academic demonstration for a B.Tech Artificial Intelligence and Machine Learning project.",
    )

    st.markdown(
        """
### CreditWise Studio

CreditWise Studio is an intelligent credit-card approval prediction system built for academic demonstration.
It combines a Streamlit FinTech dashboard with a leakage-safe scikit-learn modeling pipeline.

### Responsible AI notice

"""
    )
    st.warning(DISCLAIMER)

    st.markdown(
        """
### What the system does

1. Loads a synthetic Excel dataset (or a user-provided Excel/CSV file).
2. Profiles missing values, duplicates, dtypes, and target balance.
3. Builds preprocessing with `ColumnTransformer` + `Pipeline`:
   - numeric: median imputation + standard scaling
   - categorical: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`)
4. Trains and compares four classifiers:
   - Logistic Regression
   - Random Forest
   - Extra Trees
   - Gradient Boosting
5. Evaluates with accuracy, precision, recall, F1, macro/weighted F1, balanced accuracy, confusion matrices, and ROC-AUC where applicable.
6. Serves applicant predictions through the fitted pipeline without retraining.
7. Provides optional GridSearchCV, cross-validation, permutation importance, and SHAP explainability.

### Sensitive attributes policy

Gender, ethnicity, age, marital status, and citizenship are **not** used as model predictors.
Any columns matching those themes are excluded automatically.

### Methodology notes

- Stratified train/test splitting preserves class proportions.
- Preprocessing is fit only on training folds/data through the sklearn Pipeline to avoid leakage.
- When class imbalance is material, model selection prefers Macro F1 / Balanced Accuracy rather than Accuracy alone.
- Feature importance and SHAP values describe model association, not causal approval drivers.
- Real lending systems require legal review, fairness auditing, independent validation, and human oversight.

### Dataset

The bundled file `Credit_Card_Approval_Synthetic_1000.xlsx` is synthetic and intended only for classroom demonstration.

### Technology stack

Python, Streamlit, Pandas, NumPy, scikit-learn, Plotly, Matplotlib/Seaborn, Openpyxl, Joblib, optional SHAP, FPDF2.
"""
    )