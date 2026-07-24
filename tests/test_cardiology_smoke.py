"""Cardiology smoke task + audit-trail pipeline tests."""

import json

from configs.tasks import load_task
from src.pipeline.pipeline import ClinicalExtractionPipeline
from src.preprocessing.report_loader import REPORT_ID_KEY, REPORT_TEXT_KEY, load_reports


def _fake_llm(payload: dict):
    def _call(messages):
        return json.dumps(payload)

    return _call


def test_cardiology_smoke_task_loads():
    task = load_task("cardiology_smoke")
    assert task.name == "cardiology_smoke"
    assert task.send_full_text_when_no_evidence is True
    assert task.field_by_name("sternal_wound_infection") is not None
    assert task.field_by_name("reasoning") is not None


def test_pipeline_audit_trail_and_one_hot(monkeypatch):
    monkeypatch.setattr(
        "src.llm.llm_extraction.call_llm",
        _fake_llm(
            {
                "sternal_wound_infection": "Deep",
                "reoperation_required": "Yes",
                "information_sufficient": True,
                "reasoning": "Mediastinitis and Revisionseingriff documented.",
                "evidence_quotes": ["Tiefe sternale Wundinfektion, Mediastinitis"],
            }
        ),
    )
    task = load_task("cardiology_smoke")
    pipe = ClinicalExtractionPipeline(task)
    report = {
        REPORT_ID_KEY: "P001",
        REPORT_TEXT_KEY: (
            "[Diagnoseliste]\n\n"
            "[#1]\nTiefe sternale Wundinfektion, Mediastinitis\n\n"
            "[#2]\nRevisionseingriff wegen Infektion"
        ),
    }
    row, structured = pipe.run_report(report)
    assert row["status"] == "extracted"
    assert row["sternal_wound_infection"] == "Deep"
    assert row["reoperation_required"] == "Yes"
    assert row["reasoning"]
    assert "preprocessing" in row["stage_path"]
    assert "llm_extraction" in row["stage_path"]
    assert "guardrails" in row["stage_path"]
    assert structured["audit"]["stages"]
    assert structured["one_hot"]["sternal_wound_infection__Deep"] == 1
    assert structured["one_hot"]["sternal_wound_infection__None"] == 0
    assert structured["one_hot"]["reoperation_required__Yes"] == 1


def test_full_text_fallback_when_no_keywords(monkeypatch):
    called = {"n": 0}

    def spy(messages):
        called["n"] += 1
        user = messages[-1]["content"]
        assert "Full cleaned Diagnoseliste" in user or "CABG" in user
        return json.dumps(
            {
                "sternal_wound_infection": "None",
                "reoperation_required": "Unknown",
                "information_sufficient": False,
                "reasoning": "No wound/reoperation language found.",
                "evidence_quotes": [],
            }
        )

    monkeypatch.setattr("src.llm.llm_extraction.call_llm", spy)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P003",
        REPORT_TEXT_KEY: "[Diagnoseliste]\n\n[#1]\nElektive Bypass-Operation, stabil",
    }
    row, structured = pipe.run_report(report)
    assert called["n"] == 1
    assert row["status"] == "extracted"
    assert structured["audit"]["used_full_text_fallback"] is True


def test_example_her_file_loads_three_patients():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "her_diagnose_smoke.csv"
    records = load_reports(example)
    assert len(records) == 3
