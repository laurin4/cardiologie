"""Tests for patient-level HER Diagnoseliste loading."""

from pathlib import Path

from src.preprocessing.diagnose_loader import (
    build_patient_diagnoseliste_records,
    load_patient_diagnoseliste,
    load_patient_diagnoseliste_many,
    looks_like_her_diagnose_table,
)
from src.preprocessing.report_loader import (
    REPORT_ID_KEY,
    REPORT_TEXT_KEY,
    discover_her_diagnose_paths,
    load_reports,
)
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


def test_merge_multiple_her_files(tmp_path: Path):
    a = tmp_path / "HER_Diagnose_a.csv"
    b = tmp_path / "HER_Diagnose_vor2026.csv"
    a.write_text(
        "PatientID,Diagnose_Time,Diagnose_Value\n"
        "P100,2026-01-01,Sternale Wundinfektion\n",
        encoding="utf-8",
    )
    b.write_text(
        "PatientID,Diagnose_Time,Diagnose_Value\n"
        "P100,2025-06-01,CABG 2025\n"
        "P200,2025-07-01,Vorhofflimmern\n",
        encoding="utf-8",
    )
    records = load_patient_diagnoseliste_many([a, b])
    by_id = {r[REPORT_ID_KEY]: r for r in records}
    assert set(by_id) == {"P100", "P200"}
    text = by_id["P100"][REPORT_TEXT_KEY]
    assert "CABG 2025" in text
    assert "Sternale Wundinfektion" in text
    assert text.index("CABG 2025") < text.index("Sternale Wundinfektion")
    assert "HER_Diagnose" in by_id["P100"]["source_files"]

    via_loader = load_reports([a, b])
    assert {r[REPORT_ID_KEY] for r in via_loader} == {"P100", "P200"}


def test_discover_her_diagnose_paths(tmp_path: Path):
    (tmp_path / "HER_Diagnose_202601_202606.csv").write_text(
        "PatientID,Diagnose_Value\nP1,x\n", encoding="utf-8"
    )
    (tmp_path / "HER_Diagnose_vor2026.csv").write_text(
        "PatientID,Diagnose_Value\nP2,y\n", encoding="utf-8"
    )
    (tmp_path / "other.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    found = discover_her_diagnose_paths(tmp_path)
    names = {p.name for p in found}
    assert names == {"HER_Diagnose_202601_202606.csv", "HER_Diagnose_vor2026.csv"}
