"""Tests for evidence quote tagging and field-local reasoning."""

from src.evaluation.evidence_format import (
    columns_used_from_evidence,
    evidence_requirement_errors,
    extract_field_reasoning,
    format_structured_evidence,
    normalize_evidence_quotes,
    parse_sectioned_text,
    reasoning_for_field,
)
from configs.tasks.base import SchemaField


def test_normalize_and_format_structured_quotes():
    items = normalize_evidence_quotes(
        [
            {"column": "diag", "quote": "Schrittmacher neu implantiert am 1.1."},
            {"column": "epikrise", "quote": "Patient paced AAI 90/min."},
            {"column": "jetziges_leiden", "quote": "Bekannter SM."},
        ]
    )
    assert items[0]["column"] == "diag"
    tagged = format_structured_evidence(items)
    assert '"diag": Schrittmacher neu implantiert am 1.1.' in tagged
    assert '"epikrise": Patient paced AAI 90/min.' in tagged
    assert '"jetziges_leiden": Bekannter SM.' in tagged
    assert columns_used_from_evidence(items) == "diag; epikrise; jetziges_leiden"


def test_normalize_accepts_legacy_plain_strings():
    items = normalize_evidence_quotes(["nur ein satz"])
    assert items == [{"column": "unbekannt", "quote": "nur ein satz"}]


def test_parse_sectioned_locate():
    text = (
        "[Verlegungsbericht] FallNummer=F1\n\n"
        "[diag]\nSchrittmacher neu implantiert am 1.1.\n\n"
        "[epikrise]\nStabil, kein Fieber.\n"
    )
    sections = parse_sectioned_text(text)
    assert "diag" in sections


def test_extract_field_reasoning_from_aggregated():
    agg = (
        "### Schrittmacher\nNeu implantiert laut diag.\n\n"
        "### Vorhofflimmern\nKein neues VHF.\n"
    )
    assert "Neu implantiert" in extract_field_reasoning(agg, "pacemaker")
    assert "Kein neues VHF" in extract_field_reasoning(agg, "atrial_fibrillation")


def test_reasoning_for_field_prefers_specific_column():
    pred = {
        "pacemaker_reasoning": "Nur SM.",
        "reasoning": "### Schrittmacher\nAlt\n\n### Vorhofflimmern\nAF text",
    }
    assert reasoning_for_field(pred, "pacemaker") == "Nur SM."
    assert "AF text" in reasoning_for_field(pred, "atrial_fibrillation")


def test_evidence_requirement_errors():
    fields_ok = {
        "pacemaker": "Neu",
        "evidence_quotes": [{"column": "diag", "quote": "SM neu"}],
    }
    fields_bad = {"pacemaker": "Neu", "evidence_quotes": []}
    fields_ka = {"pacemaker": "k.A.", "evidence_quotes": []}
    schema = (
        SchemaField(name="pacemaker", type="enum", enum=("Neu", "k.A."), required=True),
        SchemaField(name="evidence_quotes", type="array", required=True, default=[]),
    )
    assert evidence_requirement_errors(fields_ok, schema) == []
    assert evidence_requirement_errors(fields_bad, schema)
    assert evidence_requirement_errors(fields_ka, schema) == []
