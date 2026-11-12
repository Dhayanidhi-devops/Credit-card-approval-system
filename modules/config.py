"""Shared configuration constants for CreditWise Studio."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DEFAULT_DATASET_PATH = DATA_DIR / "Credit_Card_Approval_Synthetic_1000.xlsx"

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_CANDIDATES = (
    "Approval_Status",
    "approval_status",
    "TARGET",
    "Target",
    "target",
    "Status",
    "status",
    "Approved",
    "Decision",
)

ID_CANDIDATES = (
    "Applicant_ID",
    "applicant_id",
    "ID",
    "Id",
    "id",
    "Application_ID",
)

# Sensitive demographic attributes must never be used as predictors.
FORBIDDEN_FEATURE_KEYWORDS = (
    "gender",
    "sex",
    "ethnicity",
    "race",
    "age",
    "dob",
    "date_of_birth",
    "marital",
    "citizenship",
    "nationality",
    "religion",
    "caste",
)

APPROVED_LABELS = {"approved", "approve", "yes", "y", "1", "true", "accepted"}
DECLINED_LABELS = {"declined", "decline", "rejected", "reject", "no", "n", "0", "false"}

NUMERIC_FEATURE_HINTS = {
    "Years_Employed": {
        "label": "Years Employed",
        "min": 0,
        "max": 50,
        "step": 1,
        "unit": "years",
    },
    "Annual_Income_INR": {
        "label": "Annual Income (INR)",
        "min": 0,
        "max": 10_000_000,
        "step": 1000,
        "unit": "INR",
    },
    "Monthly_Expenses_INR": {
        "label": "Monthly Expenses (INR)",
        "min": 0,
        "max": 1_000_000,
        "step": 500,
        "unit": "INR",
    },
    "Outstanding_Debt_INR": {
        "label": "Outstanding Debt (INR)",
        "min": 0,
        "max": 20_000_000,
        "step": 1000,
        "unit": "INR",
    },
    "Previous_Payment_Amount_INR": {
        "label": "Previous Payment Amount (INR)",
        "min": 0,
        "max": 1_000_000,
        "step": 500,
        "unit": "INR",
    },
    "Credit_Score": {
        "label": "Credit Score",
        "min": 300,
        "max": 900,
        "step": 1,
        "unit": "score",
    },
    "Credit_History_Years": {
        "label": "Credit History Years",
        "min": 0,
        "max": 50,
        "step": 1,
        "unit": "years",
    },
    "Existing_Loans_Count": {
        "label": "Existing Loans Count",
        "min": 0,
        "max": 20,
        "step": 1,
        "unit": "count",
    },
    "Credit_Utilization_Percentage": {
        "label": "Credit Utilization Percentage",
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
        "unit": "%",
    },
    "Requested_Credit_Limit_INR": {
        "label": "Requested Credit Limit (INR)",
        "min": 0,
        "max": 5_000_000,
        "step": 1000,
        "unit": "INR",
    },
}

CATEGORICAL_FEATURE_HINTS = {
    "Bank_Customer_Status": {
        "label": "Bank Customer Status",
        "options": ["Yes", "No"],
    },
    "Education_Qualification": {
        "label": "Education Qualification",
        "options": ["High School", "Diploma", "Bachelor", "Master", "Doctorate"],
    },
    "Employment_Status": {
        "label": "Employment Status",
        "options": [
            "Salaried",
            "Self-Employed",
            "Unemployed",
            "Contract",
            "Business Owner",
        ],
    },
    "Previous_Default": {
        "label": "Previous Default",
        "options": ["Yes", "No"],
    },
    "KYC_Status": {
        "label": "KYC Status",
        "options": ["Verified", "Pending", "Rejected"],
    },
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
}

METRIC_OPTIONS = {
    "macro_f1": "Macro F1-score",
    "weighted_f1": "Weighted F1-score",
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
}

DISCLAIMER = (
    "Academic demonstration only. This application uses synthetic data and must not "
    "be used for real lending, credit underwriting, or financial decisions. "
    "Predictions are model estimates, not guaranteed approval outcomes. "
    "Any real-world deployment would require legal review, fairness testing, "
    "and human oversight."
)