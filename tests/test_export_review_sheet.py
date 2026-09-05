"""Tests for the manual review sheet export."""

from pathlib import Path

from src.evaluation.review_sheet import build_review_rows, write_csv, write_excel


def test_build_review_rows_flattens_quotes():
    rows = build_review_rows(
        [
            {
                "report_id": "123",
                "patient_id": "123",
                "fall_nummers": "F9 | F10",
                "verlegung_fallnr": "F9",
                "verlegung_matched": "True",
                "status": "extracted",
                "reoperation_required": "Ja",
                "cerebrovascular_event": "Keine",
                "information_sufficient": "True",
                "evidence_quotes": '["Zitat A", "Zitat B"]',
                "reasoning": "Begründung",
            }
        ]
    )
    assert rows[0]["patient_id"] == "123"
    assert rows[0]["fall_nummers"] == "F9 | F10"
    assert rows[0]["verlegung_fallnr"] == "F9"
    assert rows[0]["verlegung_matched"] == "True"
    assert "Zitat A" in rows[0]["evidence_quotes"]
    assert rows[0]["reoperation_required"] == "Ja"
    assert rows[0]["correct_reop"] == ""
    assert "sternal_wound_infection" not in rows[0]
    assert "correct_swi" not in rows[0]


def test_write_csv(tmp_path: Path):
    out = tmp_path / "review.csv"
    rows = build_review_rows(
        [
            {
                "report_id": "1",
                "patient_id": "1",
                "fall_nummers": "F1",
                "verlegung_fallnr": "F1",
                "verlegung_matched": "True",
                "status": "extracted",
                "reoperation_required": "Nein",
                "information_sufficient": "True",
                "evidence_quotes": "[]",
                "reasoning": "ok",
            }
        ]
    )
    write_csv(rows, out)
    text = out.read_text(encoding="utf-8-sig")
    assert "patient_id;fall_nummers;verlegung_fallnr;verlegung_matched;status;" in text
    assert "sternal_wound_infection" not in text
    assert "pacemaker" in text
    assert "atrial_fibrillation" in text
    assert "reoperation_required" in text
    assert "F1" in text


def test_enrich_fall_keys_from_raw(tmp_path: Path):
    from src.evaluation.review_sheet import enrich_fall_keys_from_raw

    diag = tmp_path / "HER_Diagnose_test.csv"
    verl = tmp_path / "HER_Verlegungsbericht_test.csv"
    diag.write_text(
        "PatientID;FallNummer;Diagnose_Value;Diagnose_Time\n"
        "P001;F100;Test Diagnose;2024-01-01\n",
        encoding="utf-8",
    )
    verl.write_text(
        "patnr;FallNummer;berdat;diag;epikrise;jetziges_leiden;prozedere\n"
        "X;F100;2024-06-01;SM ok;;;\n",
        encoding="utf-8",
    )
    rows = [{"report_id": "P001", "status": "extracted"}]
    enriched = enrich_fall_keys_from_raw(
        rows, diagnose_paths=[diag], verlegung_paths=[verl]
    )
    assert enriched[0]["fall_nummers"] == "F100"
    assert enriched[0]["verlegung_fallnr"] == "F100"
    assert enriched[0]["verlegung_matched"] == "True"


def test_write_excel(tmp_path: Path):
    out = tmp_path / "review.xlsx"
    rows = build_review_rows(
        [
            {
                "report_id": "1",
                "status": "extracted",
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
