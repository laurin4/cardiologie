"""Tests for patient-level HER Diagnoseliste loading."""

from pathlib import Path

from src.preprocessing.diagnose_loader import (
    build_patient_diagnoseliste_records,
    load_patient_diagnoseliste,
    looks_like_her_diagnose_table,
)
from src.preprocessing.report_loader import REPORT_ID_KEY, REPORT_TEXT_KEY, load_reports
from src.utils.tabular_io import read_tabular


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "her_diagnose_smoke.csv"


def test_looks_like_her_diagnose():
    assert looks_like_her_diagnose_table(EXAMPLE) is True


def test_aggregates_one_row_per_patient():
    records = load_patient_diagnoseliste(EXAMPLE)
    assert len(records) == 3
    ids = {r[REPORT_ID_KEY] for r in records}
    assert ids == {"P001", "P002", "P003"}
    p001 = next(r for r in records if r[REPORT_ID_KEY] == "P001")
    assert p001["n_diagnosis_entries"] == 3
    assert "[Diagnoseliste]" in p001[REPORT_TEXT_KEY]
    assert "Mediastinitis" in p001[REPORT_TEXT_KEY]


def test_load_reports_auto_detects_her_table():
    records = load_reports(EXAMPLE)
    assert len(records) == 3
    assert all(REPORT_TEXT_KEY in r for r in records)


def test_build_from_dataframe_preserves_order():
    df = read_tabular(EXAMPLE)
    records = build_patient_diagnoseliste_records(df)
    p001 = next(r for r in records if r[REPORT_ID_KEY] == "P001")
    # Chronological: CABG mention should appear before Mediastinitis block index-wise
    text = p001[REPORT_TEXT_KEY]
    assert text.index("CABG") < text.index("Mediastinitis")
