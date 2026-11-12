"""Streamlit AppTest smoke checks for navigation + training button."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest


def main() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"App failed on load: {at.exception}"
    print("initial_run_ok")

    pages = [
        "page_scripts/dataset_explorer.py",
        "page_scripts/model_laboratory.py",
        "page_scripts/applicant_prediction.py",
        "page_scripts/model_performance.py",
        "page_scripts/prediction_history.py",
        "page_scripts/about.py",
        "page_scripts/dashboard.py",
    ]
    for page in pages:
        at.switch_page(page)
        at.run()
        assert not at.exception, f"Page {page} failed: {at.exception}"
        print(f"page_ok:{page}")

    at.switch_page("page_scripts/model_laboratory.py")
    at.run()
    assert not at.exception, at.exception

    # Click the training button and verify models land in session state.
    button = at.button(key="train_models_button")
    button.click()
    at.run()
    assert not at.exception, f"Training click failed: {at.exception}"
    bundle = at.session_state.get("training_bundle")
    assert bundle is not None and bundle.best_model_name, "No training_bundle after click"
    print("train_button_ok", bundle.best_model_name)

    at.switch_page("page_scripts/applicant_prediction.py")
    at.run()
    assert not at.exception, at.exception
    # Prediction page should no longer show the empty-models warning exclusively.
    warning_text = " ".join(w.value for w in at.warning)
    assert "No trained models available" not in warning_text
    print("prediction_page_ready")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
