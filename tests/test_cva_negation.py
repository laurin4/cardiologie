"""Negation / exclusion patterns for cardiology CVA evidence."""

from configs.tasks import load_task
from src.extraction.rule_evidence import extract_rule_evidence


def test_cva_exclusion_sentence_marks_negation():
    task = load_task("cardiology_smoke")
    cva = next(v for v in task.variables if v.name == "cerebrovascular_event")
    from configs.tasks.base import ExtractionTask

    mini = ExtractionTask(
        name=cva.name,
        fields=cva.fields,
        evidence_groups=tuple(
            g for g in task.evidence_groups if g.name in cva.evidence_group_names
        ),
        negation_patterns=task.negation_patterns,
        section_markers=task.section_markers,
        language="de",
        send_full_text_when_no_evidence=True,
    )
    text = (
        "Nach kritischer Erkrankung. "
        "Sekundär aggravierende Faktoren wie schwere metabolische Störungen "
        "sowie eine intrazerebrale Blutung respektive Ischämie konnten ausgeschlossen werden. "
        "EEG unauffällig."
    )
    ev = extract_rule_evidence(text, mini)
    assert ev["has_negation"] is True
    roles = {s.get("role") for s in ev["evidence_snippets"]}
    assert "negation" in roles
