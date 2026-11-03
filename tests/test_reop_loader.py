"""Re-OP export: loader, FallNummer attach, and pipeline source handling."""

import json

import pandas as pd
import pytest

from configs.tasks import load_task
from src.pipeline.pipeline import ClinicalExtractionPipeline
from src.preprocessing.reop_loader import (
    REOP_LOADED_KEY,
    REOP_TEXT_KEY,
    build_reop_by_fall,
    discover_her_reop_paths,
    looks_like_her_reop_table,
)
from src.preprocessing.report_loader import REPORT_ID_KEY, REPORT_TEXT_KEY, load_reports
from src.preprocessing.verlegung_loader import select_text_for_source

REOP_HEADER = (
    "FallNummer;FallKey;Index_OP_TerminNummer;Index_OPNw_Eingriff;"
    "berdat;bername;operation;zugang\n"
)
REOP_ROWS = (
    "F100;F100;41;ACBP mit HLM;2026-01-12 09:00:00.0000000;"
    "Operationsbericht Herzchirurgie;Sekundärer Thoraxverschluss;NULL\n"
    "F100;F100;41;ACBP mit HLM;2026-01-10 16:00:00.0000000;"
    "Operationsbericht Herzchirurgie;Rethorakotomie und Hämatomausräumung;Re-Sternotomie\n"
)


def _write_reop(tmp_path, name="HER_2026_08_REOPBericht.csv"):
    path = tmp_path / name
    path.write_text(REOP_HEADER + REOP_ROWS, encoding="utf-8-sig")
    return path


def _write_verlegung(tmp_path):
    path = tmp_path / "HER_Verlegungsbericht_2026_01_08.csv"
    path.write_text(
        "FallNummer;patnr;berdat;diag;epikrise;jetztleid;procedere\n"
        "F100;P1;2026-01-20;Diag A;Epi A;;\n"
        "F200;P2;2026-01-21;Diag B;Epi B;;\n",
        encoding="utf-8",
    )
    return path


def test_build_reop_by_fall_orders_by_date_and_drops_null():
    df = pd.read_csv(
        pd.io.common.StringIO(REOP_HEADER + REOP_ROWS), sep=";", dtype=str
    )
    by_fall = build_reop_by_fall(df)
    assert set(by_fall) == {"F100"}
    info = by_fall["F100"]
    text = info[REOP_TEXT_KEY]
    assert info["reop_n_ops"] == 2
    assert "[Index_OPNw_Eingriff]\nACBP mit HLM" in text
    assert text.count("ACBP mit HLM") == 1
    assert text.index("1. 2026-01-10: Rethorakotomie") < text.index(
        "2. 2026-01-12: Sekundärer Thoraxverschluss"
    )
    assert "(Zugang: Re-Sternotomie)" in text
    assert "NULL" not in text


def test_reop_table_detection_and_discovery(tmp_path):
    reop = _write_reop(tmp_path)
    # General OP-Bericht: same columns, but lists ALL OPs → must not count as Re-OP.
    other_op = tmp_path / "HER_OP_Bericht_2026_01_08.csv"
    other_op.write_text(REOP_HEADER + REOP_ROWS, encoding="utf-8")
    assert looks_like_her_reop_table(reop)
    assert not looks_like_her_reop_table(other_op)
    assert discover_her_reop_paths(tmp_path) == [reop]


def test_explicit_reop_attaches_by_fallnummer(tmp_path):
    verl = _write_verlegung(tmp_path)
    reop = _write_reop(tmp_path)
    op = tmp_path / "HER_OP_Bericht_2026_01_08.csv"
    op.write_text(REOP_HEADER + REOP_ROWS.replace("F100", "F200"), encoding="utf-8")
    records = {r["verlegung_fallnr"]: r for r in load_reports([verl, reop, op])}
    assert set(records) == {"F100", "F200"}
    assert "Rethorakotomie" in records["F100"][REOP_TEXT_KEY]
    assert records["F100"]["reop_n_ops"] == 2
    # F200 not in the Re-OP export: loaded flag set, no text → "no Re-OP"
    assert records["F200"][REOP_LOADED_KEY] is True
    assert records["F200"][REOP_TEXT_KEY] == ""
    assert select_text_for_source(records["F100"], "reop").startswith("[ReOP-Bericht]")


def test_reop_alone_is_not_a_cohort(tmp_path):
    with pytest.raises(ValueError, match="Re-OP export alone"):
        load_reports([_write_reop(tmp_path)])


_REOP_VARS = (
    "reoperation_required",
    "reoperation_context",
    "reoperation_reason",
    "rethoracotomy",
    "rethoracotomy_context",
)


def _spy_llm(seen):
    from tests.test_cardiology_smoke import _payload_for_system

    def spy(messages):
        system, user = messages[0]["content"], messages[-1]["content"]
        seen.append((system, user))
        return json.dumps(_payload_for_system(system))

    return spy


def _base_report(**extra):
    report = {
        REPORT_ID_KEY: "F100",
        REPORT_TEXT_KEY: "[Verlegungsbericht]\n\n[diag]\nRevisionseingriff",
        "diagnoseliste_text": "[Diagnoseliste]\n\n[#1]\nRevisionseingriff wegen Infektion",
        "verlegung_text": "[Verlegungsbericht]\n\n[diag]\nRevisionseingriff",
        "verlegung_fallnr": "F100",
        "fall_nummers": ["F100"],
    }
    report.update(extra)
    return report


def test_case_missing_from_loaded_reop_export_is_nein_without_llm(monkeypatch):
    seen = []
    monkeypatch.setattr("src.llm.llm_extraction.call_llm", _spy_llm(seen))
    monkeypatch.setattr("src.llm.llm_extraction.time.sleep", lambda *_: None)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    row, structured = pipe.run_report(
        _base_report(**{REOP_LOADED_KEY: True, REOP_TEXT_KEY: ""})
    )
    assert len(seen) == 10 - len(_REOP_VARS)
    assert row["status"] == "extracted"
    assert row["reoperation_required"] == "Nein"
    assert row["rethoracotomy"] == "Nein"
    assert row["reoperation_reason"] == "Keine"
    assert row["reoperation_context"] == ""
    assert row["reoperation_required_source_report"] == "ReOP-Bericht"
    rule_stages = [s for s in structured["audit"]["stages"] if s["stage"] == "source_missing_rule"]
    assert {s["variable"] for s in rule_stages} == set(_REOP_VARS)


def test_reop_text_is_sent_whole_to_reop_variables(monkeypatch):
    seen = []
    monkeypatch.setattr("src.llm.llm_extraction.call_llm", _spy_llm(seen))
    monkeypatch.setattr("src.llm.llm_extraction.time.sleep", lambda *_: None)
    df = pd.read_csv(pd.io.common.StringIO(REOP_HEADER + REOP_ROWS), sep=";", dtype=str)
    reop_info = build_reop_by_fall(df)["F100"]
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    row, _ = pipe.run_report(_base_report(**{REOP_LOADED_KEY: True, **reop_info}))
    assert len(seen) == 10
    reop_users = [u for s, u in seen if "reoperation_reason (Hauptgrund" in s]
    assert reop_users and "[Index_OPNw_Eingriff]" in reop_users[0]
    assert "2. 2026-01-12: Sekundärer Thoraxverschluss" in reop_users[0]
    assert row["reoperation_reason"] == "Wunde/Infektion/Sternum"
    assert row["rethoracotomy_source_report"] == "ReOP-Bericht"
    assert row["pacemaker_source_report"] == "Verlegungsbericht"


def test_without_reop_export_falls_back_to_previous_sources(monkeypatch):
    seen = []
    monkeypatch.setattr("src.llm.llm_extraction.call_llm", _spy_llm(seen))
    monkeypatch.setattr("src.llm.llm_extraction.time.sleep", lambda *_: None)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    row, _ = pipe.run_report(_base_report())
    assert len(seen) == 10
    assert row["reoperation_required"] == "Ja"
    assert row["reoperation_required_source_report"] == "Diagnoseliste+Verlegungsbericht"
    assert row["rethoracotomy_source_report"] == "Verlegungsbericht"
