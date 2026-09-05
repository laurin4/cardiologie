"""Cardiology smoke task + audit-trail pipeline tests."""

import json

from configs.tasks import load_task
from configs.tasks.base import ExtractionTask
from src.pipeline.pipeline import ClinicalExtractionPipeline
from src.preprocessing.report_loader import REPORT_ID_KEY, REPORT_TEXT_KEY
from src.prompts.registry import build_system_prompt


def _payload_for_system(system: str) -> dict:
    """Build a minimal valid JSON payload for whichever variable prompt is active."""
    checks = [
        (
            "pacemaker (permanenter Schrittmacher",
            {
                "pacemaker": "Kein",
                "information_sufficient": True,
                "reasoning": "Kein neuer SM.",
                "evidence_quotes": [],
            },
        ),
        (
            "atrial_fibrillation (Vorhofflimmern",
            {
                "atrial_fibrillation": "Kein",
                "information_sufficient": True,
                "reasoning": "Kein neues VHF.",
                "evidence_quotes": [],
            },
        ),
        (
            "cerebrovascular_event (neues zerebrovaskuläres",
            {
                "cerebrovascular_event": "Keine",
                "information_sufficient": True,
                "reasoning": "Kein CVA.",
                "evidence_quotes": [],
            },
        ),
        (
            "reoperation_context\n",
            {
                "reoperation_context": "Revisionseingriff wegen Infektion",
                "information_sufficient": True,
                "reasoning": "Kontext gefunden.",
                "evidence_quotes": ["Revisionseingriff"],
            },
        ),
        (
            "reoperation_required (Re-Operation erforderlich",
            {
                "reoperation_required": "Ja",
                "information_sufficient": True,
                "reasoning": "Revisionseingriff dokumentiert.",
                "evidence_quotes": ["Revisionseingriff"],
            },
        ),
        (
            "multi_system_failure (Multi-Organ-Versagen",
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
            "rethoracotomy (Re-Thorakotomie",
            {
                "rethoracotomy": "Nein",
                "information_sufficient": True,
                "reasoning": "Keine Re-Thorakotomie.",
                "evidence_quotes": [],
            },
        ),
        (
            "liver_cirrhosis (Leberzirrhose",
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
    assert len(task.variables) == 9
    assert task.field_by_name("sternal_wound_infection") is None
    assert task.field_by_name("cerebrovascular_event").enum == (
        "Keine",
        "TIA",
        "Schlaganfall",
        "Unbekannt",
        "k.A.",
    )
    assert task.field_by_name("pacemaker").enum == (
        "Neu",
        "Schon vorhanden",
        "Kein",
        "Unbekannt",
        "k.A.",
    )
    assert task.field_by_name("atrial_fibrillation").enum == (
        "Neu",
        "Vorbestehend",
        "Kein",
        "Unbekannt",
        "k.A.",
    )
    assert task.field_by_name("reoperation_context").type == "string"
    by_name = {v.name: v for v in task.variables}
    assert by_name["pacemaker"].text_source == "verlegung"
    assert by_name["atrial_fibrillation"].text_source == "both"
    assert by_name["multi_system_failure"].text_source == "verlegung"


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
    monkeypatch.setattr("src.llm.llm_extraction.time.sleep", lambda *_: None)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P001",
        REPORT_TEXT_KEY: (
            "[Diagnoseliste]\n\n"
            "[#1]\nRevisionseingriff wegen Infektion\n\n"
            "[#2]\nKein neuer Schrittmacher"
        ),
        "diagnoseliste_text": (
            "[Diagnoseliste]\n\n[#1]\nRevisionseingriff wegen Infektion"
        ),
        "verlegung_text": (
            "[Verlegungsbericht]\n\n[diag]\nKein neuer Schrittmacher\n\n"
            "[epikrise]\nStabil"
        ),
        "patient_id": "P001",
        "fall_nummers": ["F100", "F200"],
        "verlegung_fallnr": "F100",
    }
    row, structured = pipe.run_report(report)
    assert calls["n"] == 9
    assert row["status"] == "extracted"
    assert row["patient_id"] == "P001"
    assert row["fall_nummers"] == "F100 | F200"
    assert row["verlegung_fallnr"] == "F100"
    assert row["verlegung_matched"] == "True"
    assert row["reoperation_required"] == "Ja"
    assert "Revisionseingriff" in row["reoperation_context"]
    assert structured["audit"]["per_variable_llm"] is True
    assert len(structured["audit"]["llm"]["per_variable"]) == 9
    assert structured["one_hot"]["cerebrovascular_event__Keine"] == 1
    assert "sternal_wound_infection" not in row


def test_pipeline_partial_when_some_variables_fail(monkeypatch):
    def spy(messages):
        system = messages[0]["content"]
        if "pacemaker (permanenter Schrittmacher" in system:
            return ""  # force failure after retries
        return json.dumps(_payload_for_system(system))

    monkeypatch.setattr("src.llm.llm_extraction.call_llm", spy)
    monkeypatch.setattr("src.llm.llm_extraction.LLM_MAX_RETRIES", 0)
    monkeypatch.setattr("src.llm.llm_extraction.time.sleep", lambda *_: None)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P001",
        REPORT_TEXT_KEY: "[Diagnoseliste]\n\n[#1]\nRevisionseingriff",
        "diagnoseliste_text": "[Diagnoseliste]\n\n[#1]\nRevisionseingriff",
        "verlegung_text": "[Verlegungsbericht]\n\n[diag]\nRevisionseingriff",
    }
    row, _ = pipe.run_report(report)
    assert row["status"] == "partial"
    assert row["reoperation_required"] == "Ja"
    assert "pacemaker" in row["schema_errors"]


def test_full_text_fallback_still_runs_all_variables(monkeypatch):
    called = {"n": 0}

    def spy(messages):
        called["n"] += 1
        return json.dumps(_payload_for_system(messages[0]["content"]))

    monkeypatch.setattr("src.llm.llm_extraction.call_llm", spy)
    monkeypatch.setattr("src.llm.llm_extraction.time.sleep", lambda *_: None)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P001",
        REPORT_TEXT_KEY: "Keine bekannten Triggerworte in diesem Text.",
        "diagnoseliste_text": "Keine bekannten Triggerworte in diesem Text.",
        "verlegung_text": "Ebenfalls ohne Keywords.",
    }
    row, _ = pipe.run_report(report)
    assert called["n"] == 9
    assert row["status"] == "extracted"


def test_pipeline_uses_verlegung_source_for_pacemaker(monkeypatch):
    seen = {"texts": []}

    def spy(messages):
        user = messages[1]["content"]
        system = messages[0]["content"]
        if "pacemaker (permanenter Schrittmacher" in system:
            seen["texts"].append(user)
        return json.dumps(_payload_for_system(system))

    monkeypatch.setattr("src.llm.llm_extraction.call_llm", spy)
    monkeypatch.setattr("src.llm.llm_extraction.time.sleep", lambda *_: None)
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P001",
        REPORT_TEXT_KEY: "[Diagnoseliste]\nNur Diagnose",
        "diagnoseliste_text": "[Diagnoseliste]\nNur Diagnose",
        "verlegung_text": "[Verlegungsbericht]\n[diag]\nSchrittmacher implantiert",
    }
    pipe.run_report(report)
    assert seen["texts"]
    assert "Schrittmacher implantiert" in seen["texts"][0]
    assert "Nur Diagnose" not in seen["texts"][0]
