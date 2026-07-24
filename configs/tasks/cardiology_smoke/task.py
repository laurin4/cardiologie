"""
Cardiology smoke extraction task.

Smoke-test scope (expand later to the full cardiology variable list):
  1. sternal_wound_infection  -> None | Superficial | Deep
  2. reoperation_required     -> No | Yes | Unknown

Plus audit fields so every decision is showcaseable:
  - reasoning
  - evidence_quotes
  - information_sufficient  (was there enough Diagnoseliste signal?)

Input: patient-level Diagnoseliste built from HER_Diagnose_* (CSV/Excel).
"""

from __future__ import annotations

from configs.tasks.base import EvidenceGroup, ExtractionTask, SchemaField

TASK = ExtractionTask(
    name="cardiology_smoke",
    description=(
        "Smoke test: extract sternal wound infection and re-operation required "
        "from a patient Diagnoseliste, with explicit LLM reasoning."
    ),
    language="de",
    send_full_text_when_no_evidence=True,
    fields=(
        SchemaField(
            name="sternal_wound_infection",
            type="enum",
            enum=("None", "Superficial", "Deep"),
            required=True,
            default="None",
            description="Sternal wound infection severity.",
        ),
        SchemaField(
            name="reoperation_required",
            type="enum",
            enum=("No", "Yes", "Unknown"),
            required=True,
            default="Unknown",
            description="Whether any re-operation was required.",
        ),
        SchemaField(
            name="information_sufficient",
            type="boolean",
            required=True,
            default=False,
            description=(
                "True if the Diagnoseliste contained enough information to assign "
                "the clinical fields confidently; False if the text is silent/ambiguous."
            ),
        ),
        SchemaField(
            name="reasoning",
            type="string",
            required=True,
            default="",
            description="Step-by-step clinical reasoning for the assigned values.",
        ),
        SchemaField(
            name="evidence_quotes",
            type="array",
            required=True,
            default=[],
            description="Verbatim quotes from the Diagnoseliste that support the decisions.",
        ),
    ),
    evidence_groups=(
        EvidenceGroup(
            name="sternal_wound",
            role="positive",
            priority=1,
            phrases=(
                "sternal wound infection",
                "sternal wound",
                "sternal infection",
                "mediastinitis",
                "deep sternal",
                "superficial sternal",
                "sternumwundinfekt",
                "sternumwundinfektion",
                "wundinfekt sternum",
                "wundinfektion sternum",
                "sternale wundinfektion",
                "sternale infektion",
                "tiefe sternale",
                "oberflächliche sternale",
                "oberflaechliche sternale",
                "osteomyelitis sternum",
                "sternumosteomyelitis",
            ),
        ),
        EvidenceGroup(
            name="reoperation",
            role="positive",
            priority=1,
            phrases=(
                "re-operation",
                "reoperation",
                "revision surgery",
                "surgical revision",
                "return to theatre",
                "re-op",
                "revisionseingriff",
                "revisionsoperation",
                "erneute operation",
                "nochmalige operation",
                "rethorakotomie",
                "re-thorakotomie",
                "nachblutung revision",
                "tamponade revision",
            ),
        ),
        EvidenceGroup(
            name="wound_context",
            role="context",
            priority=2,
            phrases=(
                "wundinfekt",
                "wundinfektion",
                "wound infection",
                "dehiszenz",
                "dehiscence",
                "sternum",
                "sternal",
            ),
        ),
        EvidenceGroup(
            name="negation",
            role="negation",
            priority=3,
            phrases=(
                "kein wundinfekt",
                "keine wundinfektion",
                "no wound infection",
                "kein reeingriff",
                "keine revision",
                "no reoperation",
                "no re-operation",
                "ohne revision",
            ),
        ),
    ),
    section_markers=(
        ("[Diagnoseliste]", "diagnoseliste"),
    ),
    negation_patterns=(
        (r"\bkein(?:e|en)?\b.{0,30}?\bwundinfekt", "kein_wundinfekt"),
        (r"\bno\b.{0,30}?\bwound infection\b", "no_wound_infection"),
        (r"\bkein(?:e|en)?\b.{0,30}?\brevision", "keine_revision"),
        (r"\bno\b.{0,30}?\bre-?operation\b", "no_reoperation"),
    ),
    prompt_name="cardiology_smoke",
    consistency_rules=(
        {
            "type": "requires",
            "if": {"information_sufficient": True},
            "then_required": ["reasoning", "sternal_wound_infection", "reoperation_required"],
        },
    ),
)
