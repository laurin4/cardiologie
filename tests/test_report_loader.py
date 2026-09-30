from src.preprocessing.report_loader import (
    REPORT_ID_KEY,
    REPORT_TEXT_KEY,
    load_reports,
    load_reports_from_csv,
    load_reports_from_txt_dir,
)


def test_load_txt_dir(tmp_path):
    (tmp_path / "r1.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "r2.txt").write_text("world", encoding="utf-8")
    records = load_reports_from_txt_dir(tmp_path)
    assert len(records) == 2
    ids = {r[REPORT_ID_KEY] for r in records}
    assert ids == {"r1", "r2"}


def test_load_csv(tmp_path):
    p = tmp_path / "reports.csv"
    p.write_text(
        "report_id;report_text;dept\n1;fever noted;icu\n2;stable;ward\n", encoding="utf-8"
    )
    records = load_reports_from_csv(p, id_column="report_id", text_column="report_text")
    assert len(records) == 2
    assert records[0][REPORT_TEXT_KEY] == "fever noted"
    assert records[0]["dept"] == "icu"


def test_load_reports_dispatch(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    records = load_reports(tmp_path)
    assert len(records) == 1


def test_explicit_classic_verlegung_diagnose_austritt_no_year_leak(tmp_path):
    """2026-style explicit paths: classic Verlegung primary; no auto other-year files."""
    verl = tmp_path / "HER_Verlegungsbericht_2026_01_08.csv"
    verl.write_text(
        "FallNummer;patnr;berdat;diag;epikrise;jetztleid;procedere\n"
        "F100;P1;2026-01-08;SM neu;Epi;Leiden;Proc\n"
        "F200;P2;2026-01-09;kein SM;;;\n",
        encoding="utf-8",
    )
    diag = tmp_path / "HER_Diagnose_202601_202606.csv"
    diag.write_text(
        "PatientID;FallNummer;Diagnose_Value;Diagnose_Time\n"
        "P1;F100;Hypertonie;2026-01-01\n"
        "P2;F200;Diabetes;2026-01-02\n",
        encoding="utf-8",
    )
    austritt = tmp_path / "HER_Austrittsbericht_2026_01_08.csv"
    austritt.write_text(
        "FallNummer;patnr;berdat;stat_ein;anamn\n"
        "F100;P1;2026-01-10;keine Zirrhose;Anamn 2026\n",
        encoding="utf-8",
    )
    # Poison 2025 files in same folder — must NOT be attached when paths are explicit.
    (tmp_path / "HER_Austrittsbericht_2025.csv").write_text(
        "FallNummer;patnr;berdat;stat_ein;anamn\n"
        "F100;P1;2025-01-01;POISON ZIRRHOSE 2025;poison\n",
        encoding="utf-8",
    )
    (tmp_path / "HER_Diagnose_vor2026.csv").write_text(
        "PatientID;FallNummer;Diagnose_Value;Diagnose_Time\n"
        "P1;F100;POISON_DIAG_2025;2025-01-01\n",
        encoding="utf-8",
    )
    op = tmp_path / "HER_OP_Bericht_2026_01_08.csv"
    op.write_text("FallNummer;text\nF100;OP note\n", encoding="utf-8")

    records = load_reports([verl, diag, austritt, op])
    assert len(records) == 2
    by_fall = {r["verlegung_fallnr"]: r for r in records}
    assert set(by_fall) == {"F100", "F200"}

    r100 = by_fall["F100"]
    assert "SM neu" in r100["verlegung_text"]
    assert "Hypertonie" in (r100.get("diagnoseliste_text") or "")
    assert "POISON_DIAG_2025" not in (r100.get("diagnoseliste_text") or "")
    assert "keine Zirrhose" in (r100.get("austritt_text") or "")
    assert "POISON ZIRRHOSE 2025" not in (r100.get("austritt_text") or "")
    assert r100.get("input_kind") == "patient_verlegung+diagnoseliste"

    r200 = by_fall["F200"]
    assert "kein SM" in r200["verlegung_text"]
    assert "Diabetes" in (r200.get("diagnoseliste_text") or "")
    # F200 has no Austritt row in the explicit 2026 file
    assert not (r200.get("austritt_text") or "").strip()
