"""Tests for Verlegungsbericht loader and multi-source text selection."""

import pandas as pd

from src.preprocessing.verlegung_loader import (
    DIAGNOSELISTE_TEXT_KEY,
    VERLEGUNG_TEXT_KEY,
    build_latest_verlegung_by_patient,
    select_text_for_source,
)


def test_latest_verlegung_by_berdat():
    df = pd.DataFrame(
        [
            {
                "patnr": "P1",
                "berdat": "2024-01-01",
                "diag": "alt",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
            {
                "patnr": "P1",
                "berdat": "2025-06-01",
                "diag": "neu SM",
                "epikrise": "Epikrise neu",
                "jetziges_leiden": "",
                "prozedere": "Prozedere",
            },
            {
                "patnr": "P2",
                "berdat": "2024-03-01",
                "diag": "nur P2",
                "epikrise": "",
                "jetziges_leiden": "",
                "prozedere": "",
            },
        ]
    )
    by_patient = build_latest_verlegung_by_patient(df)
    assert set(by_patient) == {"P1", "P2"}
    assert "neu SM" in by_patient["P1"][VERLEGUNG_TEXT_KEY]
    assert "alt" not in by_patient["P1"][VERLEGUNG_TEXT_KEY]
    assert "[diag]" in by_patient["P1"][VERLEGUNG_TEXT_KEY]
    assert by_patient["P1"]["n_verlegung_rows"] == 2


def test_select_text_for_source_modes():
    report = {
        DIAGNOSELISTE_TEXT_KEY: "Diagnose Text",
        VERLEGUNG_TEXT_KEY: "Verlegung Text",
        "report_text": "Diagnose Text",
    }
    assert select_text_for_source(report, "diagnoseliste") == "Diagnose Text"
    assert select_text_for_source(report, "verlegung") == "Verlegung Text"
    both = select_text_for_source(report, "both")
    assert "Diagnose Text" in both and "Verlegung Text" in both
    assert select_text_for_source(report, "report") == "Diagnose Text"


def test_select_text_falls_back_to_report_text():
    report = {"report_text": "[Diagnoseliste]\nNur legacy"}
    assert "Nur legacy" in select_text_for_source(report, "diagnoseliste")
    assert select_text_for_source(report, "verlegung") == ""
