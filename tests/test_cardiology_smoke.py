"""Cardiology smoke task + audit-trail pipeline tests."""

import json

from configs.tasks import load_task
from configs.tasks.base import ExtractionTask
from src.pipeline.pipeline import ClinicalExtractionPipeline
from src.preprocessing.report_loader import REPORT_ID_KEY, REPORT_TEXT_KEY, load_reports
from src.prompts.registry import build_system_prompt


def _payload_for_system(system: str) -> dict:
    """Build a minimal valid JSON payload for whichever variable prompt is active."""
    # Order matters: more specific / unique needles first.
    checks = [
        (
            "new_permanent_pacemaker",
            {
                "new_permanent_pacemaker": "Nein",
                "information_sufficient": True,
                "reasoning": "Kein neuer SM.",
                "evidence_quotes": [],
            },
        ),
        (
            "postop_atrial_fibrillation",
            {
                "postop_atrial_fibrillation": "Nein",
                "information_sufficient": True,
                "reasoning": "Kein neues VHF.",
                "evidence_quotes": [],
            },
        ),
        (
            "cerebrovascular_event",
            {
                "cerebrovascular_event": "Keine",
                "information_sufficient": True,
                "reasoning": "Kein CVA.",
                "evidence_quotes": [],
            },
        ),
        (
            "sternal_wound_infection",
            {
                "sternal_wound_infection": "Ja",
                "information_sufficient": True,
                "reasoning": "Mediastinitis dokumentiert.",
                "evidence_quotes": ["Mediastinitis"],
            },
        ),
        (
            "reoperation_context",
            {
                "reoperation_context": "Revisionseingriff wegen Infektion",
                "information_sufficient": True,
                "reasoning": "Kontext gefunden.",
                "evidence_quotes": ["Revisionseingriff"],
            },
        ),
        (
            "reoperation_required",
            {
                "reoperation_required": "Ja",
                "information_sufficient": True,
                "reasoning": "Revisionseingriff dokumentiert.",
                "evidence_quotes": ["Revisionseingriff"],
            },
        ),
        (
            "multi_system_failure",
            {
                "multi_system_failure": "Nein",
                "information_sufficient": True,
                "reasoning": "Kein MOV.",
                "evidence_quotes": [],
            },
        ),
        (
            "rethoracotomy_context",
            {
                "rethoracotomy_context": "",
                "information_sufficient": True,
                "reasoning": "Kein Re-Thorakotomie-Kontext.",
                "evidence_quotes": [],
            },
        ),
        (
            "rethoracotomy",
            {
                "rethoracotomy": "Nein",
                "information_sufficient": True,
                "reasoning": "Keine Re-Thorakotomie.",
                "evidence_quotes": [],
            },
        ),
        (
            "liver_cirrhosis",
            {
                "liver_cirrhosis": "Nein",
                "information_sufficient": True,
                "reasoning": "Keine Zirrhose.",
                "evidence_quotes": [],
            },
        ),
    ]
    for needle, payload in checks:
        if needle in system:
            return payload
    raise AssertionError(f"No payload matched system prompt:\n{system[:500]}")


def test_cardiology_smoke_task_loads():
    task = load_task("cardiology_smoke")
    assert task.name == "cardiology_smoke"
    assert task.language == "de"
    assert len(task.variables) == 10
    assert task.field_by_name("sternal_wound_infection").enum == ("Nein", "Ja", "Unbekannt")
    assert task.field_by_name("cerebrovascular_event").enum == ("Keine", "TIA", "Schlaganfall")
    assert task.field_by_name("reoperation_context").type == "string"


def test_per_variable_prompts_load():
    task = load_task("cardiology_smoke")
    for var in task.variables:
        mini = ExtractionTask(
            name=var.name,
            fields=var.fields,
            prompt_name=var.prompt_name,
            language="de",
        )
        prompt = build_system_prompt(mini)
        assert "AUSSCHLIESSLICH ein JSON-Objekt" in prompt or "JSON" in prompt
        assert var.name in prompt or var.label.split()[0] in prompt


def test_cardiology_keywords_are_german_oriented():
    banned = {
        "sternal wound infection",
        "permanent pacemaker",
        "atrial fibrillation",
        "multi organ failure",
        "wound infection",
        "open-chest",
    }
    task = load_task("cardiology_smoke")
    phrases = {p.lower() for g in task.evidence_groups for p in g.phrases}
    assert not (phrases & banned)


def test_pipeline_one_call_per_variable(monkeypatch):
    calls = {"n": 0}

    def spy(messages):
        calls["n"] += 1
        return json.dumps(_payload_for_system(messages[0]["content"]))

    monkeypatch.setattr("src.llm.llm_extraction.call_llm", spy)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P001",
        REPORT_TEXT_KEY: (
            "[Diagnoseliste]\n\n"
            "[#1]\nTiefe sternale Wundinfektion, Mediastinitis\n\n"
            "[#2]\nRevisionseingriff wegen Infektion"
        ),
    }
    row, structured = pipe.run_report(report)
    assert calls["n"] == 10
    assert row["status"] == "extracted"
    assert row["sternal_wound_infection"] == "Ja"
    assert row["reoperation_required"] == "Ja"
    assert "Revisionseingriff" in row["reoperation_context"]
    assert structured["audit"]["per_variable_llm"] is True
    assert len(structured["audit"]["llm"]["per_variable"]) == 10
    assert structured["one_hot"]["sternal_wound_infection__Ja"] == 1
    assert structured["one_hot"]["cerebrovascular_event__Keine"] == 1


def test_full_text_fallback_still_runs_all_variables(monkeypatch):
    called = {"n": 0}

    def spy(messages):
        called["n"] += 1
        return json.dumps(_payload_for_system(messages[0]["content"]))

    monkeypatch.setattr("src.llm.llm_extraction.call_llm", spy)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P003",
        REPORT_TEXT_KEY: "[Diagnoseliste]\n\n[#1]\nElektive Bypass-Operation, stabil",
    }
    row, structured = pipe.run_report(report)
    assert called["n"] == 10
    assert row["status"] == "extracted"
    assert structured["audit"]["used_full_text_fallback"] is True


def test_example_her_file_loads_three_patients():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "her_diagnose_smoke.csv"
    records = load_reports(example)
    assert len(records) == 3
