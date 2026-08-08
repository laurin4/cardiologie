"""Tests for LLM extraction retries."""

import json

import src.llm.llm_extraction as le
from configs.tasks import load_task


def test_extract_entities_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] < 2:
            return ""  # empty → parse error → retry
        return json.dumps(
            {
                "fever_present": True,
                "fever_grade": "high_grade",
                "fever_negated": False,
                "max_temperature_c": 39.0,
            }
        )

    monkeypatch.setattr(le, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(le, "call_llm", flaky)
    monkeypatch.setattr(le.time, "sleep", lambda *_: None)

    task = load_task("demo_extraction")
    result = le.extract_entities(
        {"llm_report_text": "Patient has high fever."},
        task,
        record_id="r1",
    )
    assert calls["n"] == 2
    assert result["fields"]["fever_present"] is True
    assert result["attempts"] == 2
    assert not any(str(e).startswith("llm_extraction_failed") for e in result["schema_errors"])


def test_extract_entities_exhausted_retries(monkeypatch):
    monkeypatch.setattr(le, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(le, "call_llm", lambda messages: "not json at all")
    monkeypatch.setattr(le.time, "sleep", lambda *_: None)

    task = load_task("demo_extraction")
    result = le.extract_entities(
        {"llm_report_text": "x"},
        task,
        record_id="r1",
    )
    assert result["attempts"] == 2
    assert any(str(e).startswith("llm_extraction_failed") for e in result["schema_errors"])
