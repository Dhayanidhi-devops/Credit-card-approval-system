# CreditWise Studio

**Intelligent Credit Card Approval Prediction System**

CreditWise Studio is a complete academic FinTech web application for predicting credit-card approval outcomes using supervised machine learning. It is designed for a B.Tech Artificial Intelligence and Machine Learning project demonstration.

> **Important:** The bundled dataset is synthetic. Predictions are for academic demonstration only and must not be used for real lending or financial decisions.

---

## Features

- Professional dark-navy FinTech dashboard (Streamlit)
- Automatic loading of the default Excel dataset
- Manual Excel / CSV upload with validation
- Missing-value and duplicate detection
- Numeric / categorical feature profiling and target detection
- Four-model laboratory:
  - Logistic Regression
  - Random Forest Classifier
  - Extra Trees Classifier
  - Gradient Boosting Classifier
- Leakage-safe `sklearn` `Pipeline` + `ColumnTransformer` preprocessing
- Stratified train/test evaluation with rich metrics
- Optional cross-validation and GridSearchCV tuning
- Applicant prediction form with probability estimates
- Prediction history (session-scoped) with CSV export
- Model persistence with Joblib + compatibility checks
- Explainability: permutation importance, tree importance, optional SHAP
- PDF / CSV prediction report downloads

---

## Technology Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ (tested on 3.11/3.12) |
| UI | Streamlit |
| Data | Pandas, NumPy, Openpyxl |
| ML | Scikit-learn, Joblib |
| Charts | Plotly, Matplotlib, Seaborn |
| Reports | FPDF2 |
| Explainability | SHAP (optional) |

---

## Project Architecture

```text
CreditWise_Studio/
├── app.py                 # Streamlit entrypoint + st.navigation
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Credit_Card_Approval_Synthetic_1000.xlsx
├── models/                # Saved Joblib artifacts
├── assets/
│   └── style.css
├── modules/               # Data, ML, explainability, reports
├── views/                 # Page render logic
├── page_scripts/          # Thin Streamlit page entrypoints for st.navigation
└── scripts/
    └── smoke_apptest.py   # AppTest smoke checks
```

`st.navigation` + `page_scripts/` replaces Streamlit’s automatic `pages/` folder so the app keeps one branded sidebar while preserving reliable multipage URL routes (`/dashboard`, `/model-laboratory`, etc.).

---

## Dataset Description

Default file: `data/Credit_Card_Approval_Synthetic_1000.xlsx`

Synthetic applicant attributes include:

- Employment / income / expense / debt fields
- Credit score, credit history, utilization, existing loans
- Requested credit limit
- Bank customer status, education, employment status
- Previous default and KYC status
- Target: `Approval_Status` (`Approved` / `Declined`)

Sensitive demographic attributes (gender, ethnicity, age, marital status, citizenship) are excluded from modeling.

---

## Machine Learning Algorithms

1. **Logistic Regression** — linear baseline with class balancing  
2. **Random Forest** — bagged decision trees  
3. **Extra Trees** — randomized ensemble trees  
4. **Gradient Boosting** — sequential boosted trees  

Preprocessing:

- Numeric: median imputation + `StandardScaler`
- Categorical: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`
- Fit only on training data through the Pipeline

Best-model selection defaults to **Macro F1** (imbalance-aware), with user-selectable metrics.

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
.source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## Windows VS Code Setup

1. Open the project folder in VS Code.
2. Open a PowerShell terminal in VS Code.
3. Create the virtual environment:

```powershell
python -m venv .venv
```

4. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. (Optional) Select the `.venv` interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter**.

6. Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

7. Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## How to Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or after activating the virtual environment:

```bash
streamlit run app.py
```

---

## How to Upload a Dataset

1. Open **Dataset Explorer**.
2. Upload an Excel (`.xlsx`) or CSV (`.csv`) file.
3. Click **Use uploaded file**.
4. Review schema warnings, missing values, and target detection.
5. The original file on disk is never overwritten.

Required for training: a target column such as `Approval_Status`.

---

## How to Train Models

1. Open **Model Laboratory**.
2. Select one or more algorithms.
3. Choose the preferred evaluation metric (Macro F1 recommended under imbalance).
4. Optionally enable GridSearchCV and/or cross-validation.
5. Click **Train Selected / All Models**.
6. Inspect metrics, confusion matrices, ROC / PR curves, and train-vs-test comparison.
7. Save the preferred model with Joblib.

---

## How to Make Predictions

1. Train models (or load a trusted local artifact).
2. Open **Applicant Prediction**.
3. Select a trained model.
4. Enter applicant attributes and click **Predict Approval Outcome**.
5. Review the predicted class, probability estimates, and explanation.
6. Download the prediction report as CSV or PDF.
7. Find the record later under **Prediction History**.

Probability outputs are model estimates on synthetic academic data — not guaranteed real-world approval probabilities.

---

## Screenshots

Add screenshots of the following pages after running locally:

1. Dashboard KPIs and charts  
2. Applicant Prediction report  
3. Model Laboratory comparison  
4. Dataset Explorer profiling  
5. Model Performance / explainability  

Suggested folder: `assets/screenshots/`

---

## Responsible AI & Security

- Synthetic data only for demonstration
- No use of gender, ethnicity, age, marital status, or citizenship as predictors
- Predictions are not financial advice or guaranteed decisions
- Saved models are loaded only when the user explicitly requests them
- Real deployment would require legal review, fairness testing, independent validation, and human oversight

---

## Limitations

- Dataset is synthetic and may not reflect real credit portfolios
- Model quality depends on uploaded data quality and schema alignment
- SHAP explainability is optional and may be unavailable on some systems
- Prediction history is session-scoped (not a production database)
- Not a substitute for regulated credit decisioning systems

---

## Future Scope

- Fairness metrics and bias audits
- Persistent database-backed applicant history
- Advanced calibration and threshold optimization
- Deployment packaging (Docker) for demos
- Broader model zoo and AutoML baselines

---

## License / Academic Use

Created for academic demonstration purposes. Use responsibly.
