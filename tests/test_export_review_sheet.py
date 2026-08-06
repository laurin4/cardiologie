"""Tests for the manual review sheet export."""

from pathlib import Path

from src.evaluation.review_sheet import build_review_rows, write_csv, write_excel


def test_build_review_rows_flattens_quotes():
    rows = build_review_rows(
        [
            {
                "report_id": "123",
                "status": "extracted",
                "sternal_wound_infection": "Tief",
                "reoperation_required": "Ja",
                "information_sufficient": "True",
                "evidence_quotes": '["Zitat A", "Zitat B"]',
                "reasoning": "Begründung",
            }
        ]
    )
    assert rows[0]["patient_id"] == "123"
    assert "Zitat A" in rows[0]["evidence_quotes"]
    assert rows[0]["correct_swi"] == ""
    assert rows[0]["correct_reop"] == ""
    assert rows[0]["notes"] == ""


def test_write_csv(tmp_path: Path):
    out = tmp_path / "review.csv"
    rows = build_review_rows(
        [
            {
                "report_id": "1",
                "status": "extracted",
                "sternal_wound_infection": "Nein",
                "reoperation_required": "Nein",
                "information_sufficient": "True",
                "evidence_quotes": "[]",
                "reasoning": "ok",
            }
        ]
    )
    write_csv(rows, out)
    text = out.read_text(encoding="utf-8-sig")
    assert "patient_id;status;" in text
    assert "1;extracted;Nein;Nein" in text


def test_write_excel(tmp_path: Path):
    out = tmp_path / "review.xlsx"
    rows = build_review_rows(
        [
            {
                "report_id": "1",
                "status": "extracted",
                "sternal_wound_infection": "Nein",
                "reoperation_required": "Nein",
                "information_sufficient": "True",
                "evidence_quotes": "[]",
                "reasoning": "ok",
            }
        ]
    )
    write_excel(rows, out)
    assert out.exists()
    assert out.stat().st_size > 0
