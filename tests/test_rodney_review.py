"""Tests for Rodney 25/variable review export."""

from src.evaluation.rodney_review import (
    build_long_rows,
    filter_dendrite_overlap,
    sample_per_variable,
    write_rodney_csv,
    write_rodney_excel,
)


def test_sample_patients_equal_variable_counts():
    from src.evaluation.rodney_review import build_long_rows, sample_patients

    rows = []
    for i in range(40):
        rows.append(
            {
                "patient_id": f"P{i}",
                "verlegung_fallnr": f"F{i}",
                "fall_nummers": f"F{i}",
                "status": "extracted",
                "pacemaker": "Neu" if i % 2 == 0 else "Kein",
                "atrial_fibrillation": "Kein",
                "liver_cirrhosis": "Nein",
            }
        )
    picked = sample_patients(rows, n_patients=25, seed=7)
    assert len(picked) == 25
    long_rows = build_long_rows(
        picked, variables=["pacemaker", "atrial_fibrillation", "liver_cirrhosis"]
    )
    counts = {}
    for r in long_rows:
        counts[r["variable"]] = counts.get(r["variable"], 0) + 1
    assert counts == {
        "pacemaker": 25,
        "atrial_fibrillation": 25,
        "liver_cirrhosis": 25,
    }


def test_sample_per_variable_caps_and_stratifies():
    long_rows = []
    for i in range(40):
        long_rows.append(
            {
                "variable": "pacemaker",
                "patient_id": f"P{i}",
                "prediction": "Neu" if i % 2 == 0 else "Kein",
                "source_report": "Verlegungsbericht",
                "source_columns": "diag",
                "evidence_quotes": "q",
                "reasoning": "r",
                "status": "extracted",
                "fall_nummers": f"F{i}",
                "verlegung_fallnr": f"F{i}",
                "correct": "",
                "notes": "",
            }
        )
    for i in range(10):
        long_rows.append(
            {
                "variable": "liver_cirrhosis",
                "patient_id": f"C{i}",
                "prediction": "Nein",
                "source_report": "Austrittsbericht",
                "source_columns": "stat_ein",
                "evidence_quotes": "",
                "reasoning": "",
                "status": "extracted",
                "fall_nummers": f"C{i}",
                "verlegung_fallnr": f"C{i}",
                "correct": "",
                "notes": "",
            }
        )
    sampled = sample_per_variable(long_rows, n_per_var=25, seed=1)
    pm = [r for r in sampled if r["variable"] == "pacemaker"]
    ci = [r for r in sampled if r["variable"] == "liver_cirrhosis"]
    assert len(pm) == 25
    assert len(ci) == 10
    assert {r["prediction"] for r in pm} == {"Neu", "Kein"}


def test_build_long_rows_uses_provenance_columns():
    rows = build_long_rows(
        [
            {
                "patient_id": "P1",
                "fall_nummers": "F1",
                "verlegung_fallnr": "F1",
                "status": "extracted",
                "pacemaker": "Neu",
                "pacemaker_source_report": "Verlegungsbericht",
                "pacemaker_source_columns": "diag, epikrise",
                "pacemaker_evidence_quotes": '["SM neu"]',
                "pacemaker_reasoning": "Neu implantiert",
                "liver_cirrhosis": "Nein",
            }
        ],
        variables=["pacemaker", "liver_cirrhosis"],
    )
    assert len(rows) == 2
    pm = next(r for r in rows if r["variable"] == "pacemaker")
    assert pm["prediction"] == "Neu"
    assert pm["source_report"] == "Verlegungsbericht"
    assert "SM neu" in pm["evidence_quotes"]
    ci = next(r for r in rows if r["variable"] == "liver_cirrhosis")
    assert ci["source_report"] == "Austrittsbericht"


def test_filter_dendrite_overlap():
    kept = filter_dendrite_overlap(
        [
            {"fall_nummers": "F100 | F101", "verlegung_fallnr": "F100"},
            {"fall_nummers": "F999", "verlegung_fallnr": "F999"},
        ],
        {"F100"},
    )
    assert len(kept) == 1
    assert "F100" in kept[0]["fall_nummers"]


def test_write_rodney_files(tmp_path):
    rows = [
        {
            "variable": "pacemaker",
            "patient_id": "P1",
            "fall_nummers": "F1",
            "verlegung_fallnr": "F1",
            "prediction": "Neu",
            "source_report": "Verlegungsbericht",
            "source_columns": "diag",
            "evidence_quotes": "SM",
            "reasoning": "ok",
            "status": "extracted",
            "correct": "",
            "notes": "",
        }
    ]
    csv_path = tmp_path / "r.csv"
    xlsx_path = tmp_path / "r.xlsx"
    write_rodney_csv(rows, csv_path)
    write_rodney_excel(rows, xlsx_path)
    assert "source_report" in csv_path.read_text(encoding="utf-8-sig")
    assert xlsx_path.exists() and xlsx_path.stat().st_size > 0
