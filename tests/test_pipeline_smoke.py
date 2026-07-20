import json

from configs.tasks import load_task
from src.pipeline.pipeline import ClinicalExtractionPipeline
from src.preprocessing.report_loader import REPORT_ID_KEY, REPORT_TEXT_KEY


def _fake_llm_factory():
    def fake_call_llm(messages):
        user = messages[-1]["content"].lower()
        if "afebrile" in user or "no fever" in user:
            return json.dumps(
                {"fever_present": False, "fever_grade": "none", "fever_negated": True,
                 "max_temperature_c": None}
            )
        grade = "low_grade" if "low grade" in user else "high_grade"
        return json.dumps(
            {"fever_present": True, "fever_grade": grade, "fever_negated": False,
             "max_temperature_c": 39.1}
        )

    return fake_call_llm


def test_pipeline_positive(monkeypatch):
    monkeypatch.setattr("src.llm.llm_extraction.call_llm", _fake_llm_factory())
    pipe = ClinicalExtractionPipeline(load_task("demo_extraction"))
    report = {
        REPORT_ID_KEY: "r1",
        REPORT_TEXT_KEY: "[Assessment]\nPatient febrile, high fever, temperature 39.1 C.",
    }
    row, structured = pipe.run_report(report)
    assert row["status"] == "extracted"
    assert row["fever_present"] is True
    assert row["fever_grade"] == "high_grade"
    assert structured["fields"]["fever_present"] is True


def test_pipeline_negation(monkeypatch):
    monkeypatch.setattr("src.llm.llm_extraction.call_llm", _fake_llm_factory())
    pipe = ClinicalExtractionPipeline(load_task("demo_extraction"))
    report = {
        REPORT_ID_KEY: "r2",
        REPORT_TEXT_KEY: "[Assessment]\nPatient is afebrile. No fever. Temperature 36.7 C.",
    }
    row, _ = pipe.run_report(report)
    assert row["fever_present"] is False
    assert row["fever_negated"] is True
    assert row["fever_grade"] == "none"


def test_pipeline_skips_when_no_evidence(monkeypatch):
    called = {"n": 0}

    def spy(messages):
        called["n"] += 1
        return "{}"

    monkeypatch.setattr("src.llm.llm_extraction.call_llm", spy)
    pipe = ClinicalExtractionPipeline(load_task("demo_extraction"))
    report = {REPORT_ID_KEY: "r3", REPORT_TEXT_KEY: "[History]\nPatient reports mild back pain."}
    row, _ = pipe.run_report(report)
    assert row["status"] == "skipped_no_evidence"
    assert called["n"] == 0
    assert row["fever_present"] is False


def test_fieldnames_include_task_fields():
    pipe = ClinicalExtractionPipeline(load_task("demo_extraction"))
    names = pipe.fieldnames()
    assert "fever_present" in names
    assert "report_id" in names
    assert "manual_review_candidate" in names
