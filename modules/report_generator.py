"""Prediction and performance report export helpers."""

from __future__ import annotations

import csv
from io import BytesIO, StringIO
from typing import Any

import pandas as pd

from modules.config import DISCLAIMER


def prediction_to_csv_bytes(prediction_result: dict[str, Any], reference: str) -> bytes:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Field", "Value"])
    writer.writerow(["Reference", reference])
    writer.writerow(["Prediction", prediction_result.get("prediction")])
    writer.writerow(["Model", prediction_result.get("model_name")])
    writer.writerow(["Timestamp", prediction_result.get("timestamp")])
    writer.writerow(["Approval Probability (model estimate)", prediction_result.get("approval_probability")])
    probs = prediction_result.get("probabilities") or {}
    for label, value in probs.items():
        writer.writerow([f"Probability::{label}", value])
    inputs = prediction_result.get("input") or {}
    for key, value in inputs.items():
        writer.writerow([f"Input::{key}", value])
    writer.writerow(["Disclaimer", DISCLAIMER])
    return buffer.getvalue().encode("utf-8")


def _pdf_safe(text: Any) -> str:
    """Normalize text for core PDF fonts (latin-1 compatible)."""
    raw = str(text)
    return raw.encode("latin-1", errors="replace").decode("latin-1")


def prediction_to_pdf_bytes(prediction_result: dict[str, Any], reference: str) -> bytes:
    try:
        from fpdf import FPDF
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PDF export requires the fpdf2 package.") from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_x(15)

    pdf.set_fill_color(11, 31, 58)
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_text_color(248, 250, 252)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 18, _pdf_safe("CreditWise Studio Prediction Report"), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_safe(f"Reference: {reference}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, _pdf_safe(f"Prediction: {prediction_result.get('prediction')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, _pdf_safe(f"Model: {prediction_result.get('model_name')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, _pdf_safe(f"Timestamp: {prediction_result.get('timestamp')}"), new_x="LMARGIN", new_y="NEXT")

    approval_p = prediction_result.get("approval_probability")
    if approval_p is not None:
        pdf.cell(
            0,
            7,
            _pdf_safe(
                f"Approval probability (model estimate only): {float(approval_p):.2%}"
            ),
            new_x="LMARGIN",
            new_y="NEXT",
        )

    probs = prediction_result.get("probabilities") or {}
    if probs:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Class Probabilities", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        for label, value in probs.items():
            pdf.cell(
                0,
                6,
                _pdf_safe(f"- {label}: {float(value):.2%}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Applicant Inputs", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    for key, value in (prediction_result.get("input") or {}).items():
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_width, 5, _pdf_safe(f"{key}: {value}"))

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width, 5, "Disclaimer")
    pdf.set_font("Helvetica", size=9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width, 5, _pdf_safe(DISCLAIMER))

    out = BytesIO()
    pdf.output(out)
    return out.getvalue()


def metrics_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def cleaned_dataset_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export a copy of the current in-memory dataset without modifying the original file."""
    return df.to_csv(index=False).encode("utf-8")