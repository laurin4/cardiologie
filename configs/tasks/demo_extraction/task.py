"""
Demo extraction task: fever documentation.

A deliberately small, neutral example (NOT a real clinical target and unrelated to
any planned cardiology variable). It exists so the framework runs and tests pass
end-to-end, and to serve as a copy-paste template for real tasks:

    1. Declare the output schema (``fields``).
    2. Declare rule-based ``evidence_groups`` (+ optional ``negation_patterns``).
    3. Point ``prompt_name`` at a template in ``prompts/``.
    4. Declare deterministic ``consistency_rules`` (guardrails).
"""

from __future__ import annotations

from configs.tasks.base import EvidenceGroup, ExtractionTask, SchemaField

TASK = ExtractionTask(
    name="demo_extraction",
    description="Neutrales Beispiel: Fieberdokumentation aus einem Bericht extrahieren.",
    language="de",
    fields=(
        SchemaField(
            name="fever_present",
            type="boolean",
            required=True,
            default=False,
            description="Ob dokumentiertes Fieber im Bericht vorliegt.",
        ),
        SchemaField(
            name="fever_grade",
            type="enum",
            enum=("keine", "niedrig", "hoch"),
            default="keine",
            description="Dokumentierter Schweregrad des Fiebers.",
        ),
        SchemaField(
            name="fever_negated",
            type="boolean",
            default=False,
            description="Ob Fieber explizit verneint/ausgeschlossen ist.",
        ),
        SchemaField(
            name="max_temperature_c",
            type="number",
            default=None,
            description="Höchste dokumentierte Temperatur in Celsius, falls angegeben.",
        ),
    ),
    evidence_groups=(
        EvidenceGroup(
            name="fever_terms",
            role="positive",
            priority=1,
            phrases=(
                "high fever",
                "low grade fever",
                "fever",
                "febrile",
                "pyrexia",
                "hyperthermia",
                "temperature",
            ),
        ),
        EvidenceGroup(
            name="associated_symptoms",
            role="context",
            priority=2,
            phrases=("chills", "rigors", "night sweats", "sweating"),
        ),
        EvidenceGroup(
            name="fever_negation",
            role="negation",
            priority=3,
            phrases=("afebrile", "no fever", "without fever", "denies fever"),
        ),
    ),
    section_markers=(
        ("[History]", "history"),
        ("[Assessment]", "assessment"),
        ("[Plan]", "plan"),
    ),
    negation_patterns=(
        (r"\bno\b.{0,20}?\bfever\b", "no_fever"),
        (r"\bwithout\b.{0,20}?\bfever\b", "without_fever"),
    ),
    prompt_name="demo_extraction",
    consistency_rules=(
        {"type": "conditional_set", "if": {"fever_negated": True}, "set": {"fever_present": False}},
        {"type": "conditional_set", "if": {"fever_present": False}, "set": {"fever_grade": "keine"}},
        {
            "type": "normalize",
            "field": "fever_grade",
            "map": {
                "none": "keine",
                "low_grade": "niedrig",
                "high_grade": "hoch",
            },
        },
        {"type": "impossible_combination", "when": {"fever_present": True, "fever_negated": True}},
        {"type": "requires", "if": {"fever_present": True}, "then_required": ["fever_grade"]},
    ),
)
