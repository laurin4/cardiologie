from configs.tasks import load_task
from src.extraction.rule_evidence import (
    METHOD_NO_EVIDENCE,
    extract_rule_evidence,
    llm_should_receive_evidence,
)


def _task():
    return load_task("demo_extraction")


def test_positive_evidence_detected():
    text = "[Assessment]\nPatient is febrile with high fever, temperature 39.1 C."
    ev = extract_rule_evidence(text, _task())
    assert ev["has_positive_evidence"] is True
    assert ev["evidence_flags"]["fever_terms"] is True
    assert ev["keyword_hits_count"] >= 1
    assert ev["llm_report_text_length"] > 0
    assert llm_should_receive_evidence(ev["evidence_snippets"]) is True


def test_negation_only_skips_llm():
    text = "[Assessment]\nPatient is afebrile. No fever documented."
    ev = extract_rule_evidence(text, _task())
    assert ev["has_negation"] is True
    assert ev["has_positive_evidence"] is False
    assert ev["reduction_method"] == METHOD_NO_EVIDENCE


def test_section_labelling():
    text = "[History]\nfeels warm\n\n[Assessment]\nhigh fever noted"
    ev = extract_rule_evidence(text, _task())
    sections = {s["section"] for s in ev["evidence_snippets"]}
    assert "assessment" in sections


def test_empty_text_returns_empty_bundle():
    ev = extract_rule_evidence("", _task())
    assert ev["has_positive_evidence"] is False
    assert ev["keyword_hits_count"] == 0
