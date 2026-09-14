"""Tests for evidence quote tagging and field-local reasoning."""

from src.evaluation.evidence_format import (
    extract_field_reasoning,
    format_tagged_evidence,
    parse_sectioned_text,
    reasoning_for_field,
)


def test_parse_sectioned_and_tag_quotes():
    text = (
        "[Verlegungsbericht] FallNummer=F1\n\n"
        "[diag]\nSchrittmacher neu implantiert am 1.1.\n\n"
        "[epikrise]\nStabil, kein Fieber.\n"
    )
    sections = parse_sectioned_text(text)
    assert "diag" in sections
    tagged = format_tagged_evidence(
        ["Schrittmacher neu implantiert am 1.1."],
        source_text=text,
        fallback_column="diag",
    )
    assert tagged.startswith('diagnose: "')
    assert "Schrittmacher neu" in tagged


def test_extract_field_reasoning_from_aggregated():
    agg = (
        "### Schrittmacher\nNeu implantiert laut diag.\n\n"
        "### Vorhofflimmern\nKein neues VHF.\n"
    )
    assert "Neu implantiert" in extract_field_reasoning(agg, "pacemaker")
    assert "Kein neues VHF" in extract_field_reasoning(agg, "atrial_fibrillation")
    assert extract_field_reasoning(agg, "pacemaker") != extract_field_reasoning(
        agg, "atrial_fibrillation"
    )


def test_reasoning_for_field_prefers_specific_column():
    pred = {
        "pacemaker_reasoning": "Nur SM.",
        "reasoning": "### Schrittmacher\nAlt\n\n### Vorhofflimmern\nAF text",
    }
    assert reasoning_for_field(pred, "pacemaker") == "Nur SM."
    assert "AF text" in reasoning_for_field(pred, "atrial_fibrillation")
    assert "Alt" not in reasoning_for_field(pred, "atrial_fibrillation")
