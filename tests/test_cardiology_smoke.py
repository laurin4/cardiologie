"""Cardiology smoke task + audit-trail pipeline tests."""

import json

from configs.tasks import load_task
from src.pipeline.pipeline import ClinicalExtractionPipeline
from src.preprocessing.report_loader import REPORT_ID_KEY, REPORT_TEXT_KEY, load_reports
from src.prompts.registry import build_system_prompt


def _fake_llm_router(payloads_by_needle: dict):
    """Return JSON based on which variable prompt needle appears in the system message."""

    def _call(messages):
        system = messages[0]["content"]
        for needle, payload in payloads_by_needle.items():
            if needle in system:
                return json.dumps(payload)
        raise AssertionError(f"No payload matched system prompt:\n{system[:400]}")

    return _call


def test_cardiology_smoke_task_loads():
    task = load_task("cardiology_smoke")
    assert task.name == "cardiology_smoke"
    assert task.language == "de"
    assert task.send_full_text_when_no_evidence is True
    assert len(task.variables) == 2
    assert task.field_by_name("sternal_wound_infection").enum == ("Nein", "Ja", "Unbekannt")
    assert task.field_by_name("reoperation_required").enum == ("Nein", "Ja", "Unbekannt")


def test_per_variable_prompts_are_german():
    task = load_task("cardiology_smoke")
    for var in task.variables:
        from configs.tasks.base import ExtractionTask

        mini = ExtractionTask(
            name=var.name,
            fields=var.fields,
            prompt_name=var.prompt_name,
            language="de",
        )
        prompt = build_system_prompt(mini)
        assert "AUSSCHLIESSLICH ein JSON-Objekt" in prompt
        assert "Nein" in prompt and "Ja" in prompt and "Unbekannt" in prompt
        assert "Oberflächlich" not in prompt
        assert "Standard:" in prompt


def test_cardiology_keywords_are_german_oriented():
    banned = {
        "sternal wound infection",
        "sternal wound",
        "sternal infection",
        "deep sternal",
        "superficial sternal",
        "wound infection",
        "revision surgery",
        "surgical revision",
        "return to theatre",
        "re-op",
        "no wound infection",
        "no reoperation",
        "no re-operation",
        "dehiscence",
        "open chest",
        "open-chest",
    }
    task = load_task("cardiology_smoke")
    phrases = {p.lower() for g in task.evidence_groups for p in g.phrases}
    assert not (phrases & banned)
    assert "sternale wundinfektion" in phrases
    assert "revisionseingriff" in phrases or "wundrevision" in phrases


def test_pipeline_two_llm_calls_and_one_hot(monkeypatch):
    calls = {"n": 0}

    def spy(messages):
        calls["n"] += 1
        system = messages[0]["content"]
        if "sternal_wound_infection" in system and "reoperation_required" not in system:
            return json.dumps(
                {
                    "sternal_wound_infection": "Ja",
                    "information_sufficient": True,
                    "reasoning": "Mediastinitis dokumentiert.",
                    "evidence_quotes": ["Mediastinitis"],
                }
            )
        if "reoperation_required" in system:
            return json.dumps(
                {
                    "reoperation_required": "Ja",
                    "information_sufficient": True,
                    "reasoning": "Revisionseingriff dokumentiert.",
                    "evidence_quotes": ["Revisionseingriff"],
                }
            )
        raise AssertionError(system[:300])

    monkeypatch.setattr("src.llm.llm_extraction.call_llm", spy)
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
    assert calls["n"] == 2
    assert row["status"] == "extracted"
    assert row["sternal_wound_infection"] == "Ja"
    assert row["reoperation_required"] == "Ja"
    assert "Mediastinitis" in row["reasoning"]
    assert "Revisionseingriff" in row["reasoning"]
    assert structured["audit"]["per_variable_llm"] is True
    assert len(structured["audit"]["llm"]["per_variable"]) == 2
    assert structured["one_hot"]["sternal_wound_infection__Ja"] == 1
    assert structured["one_hot"]["sternal_wound_infection__Nein"] == 0
    assert structured["one_hot"]["reoperation_required__Ja"] == 1


def test_english_and_old_severity_aliases_normalize_to_yn(monkeypatch):
    monkeypatch.setattr(
        "src.llm.llm_extraction.call_llm",
        _fake_llm_router(
            {
                "sternal_wound_infection (sternale": {
                    "sternal_wound_infection": "Deep",
                    "information_sufficient": True,
                    "reasoning": "Tiefe Infektion.",
                    "evidence_quotes": ["Mediastinitis"],
                },
                "reoperation_required (Re-Operation": {
                    "reoperation_required": "Yes",
                    "information_sufficient": True,
                    "reasoning": "Revision.",
                    "evidence_quotes": ["Revisionseingriff"],
                },
            }
        ),
    )
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P001",
        REPORT_TEXT_KEY: "[Diagnoseliste]\n\n[#1]\nMediastinitis, Revisionseingriff",
    }
    row, _ = pipe.run_report(report)
    assert row["sternal_wound_infection"] == "Ja"
    assert row["reoperation_required"] == "Ja"


def test_full_text_fallback_calls_both_variables(monkeypatch):
    called = {"n": 0}

    def spy(messages):
        called["n"] += 1
        user = messages[-1]["content"]
        assert "bereinigte Diagnoseliste" in user or "Bypass" in user
        system = messages[0]["content"]
        if "sternal_wound_infection" in system and "reoperation_required" not in system:
            return json.dumps(
                {
                    "sternal_wound_infection": "Unbekannt",
                    "information_sufficient": False,
                    "reasoning": "Kein klarer Wundinfekt.",
                    "evidence_quotes": [],
                }
            )
        return json.dumps(
            {
                "reoperation_required": "Unbekannt",
                "information_sufficient": False,
                "reasoning": "Keine klare Revision.",
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
    assert called["n"] == 2
    assert row["status"] == "extracted"
    assert structured["audit"]["used_full_text_fallback"] is True


def test_example_her_file_loads_three_patients():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "her_diagnose_smoke.csv"
    records = load_reports(example)
    assert len(records) == 3
