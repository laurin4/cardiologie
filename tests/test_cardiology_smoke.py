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
    assert task.language == "de"
    assert task.send_full_text_when_no_evidence is True
    assert task.field_by_name("sternal_wound_infection").enum == (
        "Keine",
        "Oberflächlich",
        "Tief",
    )


def test_prompts_and_schema_block_are_german():
    from src.prompts.registry import build_system_prompt

    prompt = build_system_prompt(load_task("cardiology_smoke"))
    assert "klinisches Informations-Extraktionssystem" in prompt
    assert "AUSSCHLIESSLICH ein JSON-Objekt" in prompt
    assert "None" not in prompt or "Keine" in prompt
    assert "Oberflächlich" in prompt
    assert "Unbekannt" in prompt
    assert "Standard:" in prompt


def test_cardiology_keywords_are_german_oriented():
    """Rule phrases must not rely on English clinical wording for matching."""
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
    assert {g.name for g in task.evidence_groups} == {
        "sternale_wundinfektion",
        "reeingriff",
        "wund_kontext",
        "verneinung",
    }


def test_pipeline_audit_trail_and_one_hot(monkeypatch):
    monkeypatch.setattr(
        "src.llm.llm_extraction.call_llm",
        _fake_llm(
            {
                "sternal_wound_infection": "Tief",
                "reoperation_required": "Ja",
                "information_sufficient": True,
                "reasoning": "Mediastinitis und Revisionseingriff sind dokumentiert.",
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
    assert row["sternal_wound_infection"] == "Tief"
    assert row["reoperation_required"] == "Ja"
    assert "dokumentiert" in row["reasoning"]
    assert "preprocessing" in row["stage_path"]
    assert structured["one_hot"]["sternal_wound_infection__Tief"] == 1
    assert structured["one_hot"]["sternal_wound_infection__Keine"] == 0
    assert structured["one_hot"]["reoperation_required__Ja"] == 1


def test_english_enum_aliases_normalized_to_german(monkeypatch):
    monkeypatch.setattr(
        "src.llm.llm_extraction.call_llm",
        _fake_llm(
            {
                "sternal_wound_infection": "Deep",
                "reoperation_required": "Yes",
                "information_sufficient": True,
                "reasoning": "Tiefe Infektion erkannt.",
                "evidence_quotes": ["Mediastinitis"],
            }
        ),
    )
    pipe = ClinicalExtractionPipeline(load_task("cardiology_smoke"))
    report = {
        REPORT_ID_KEY: "P001",
        REPORT_TEXT_KEY: "[Diagnoseliste]\n\n[#1]\nMediastinitis, Revisionseingriff",
    }
    row, _ = pipe.run_report(report)
    assert row["sternal_wound_infection"] == "Tief"
    assert row["reoperation_required"] == "Ja"


def test_full_text_fallback_when_no_keywords(monkeypatch):
    called = {"n": 0}

    def spy(messages):
        called["n"] += 1
        user = messages[-1]["content"]
        assert "bereinigte Diagnoseliste" in user or "CABG" in user
        assert "Klinische Evidenz zur Extraktion" in user
        return json.dumps(
            {
                "sternal_wound_infection": "Keine",
                "reoperation_required": "Unbekannt",
                "information_sufficient": False,
                "reasoning": "Keine Hinweise auf Wundinfekt oder Revision gefunden.",
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
