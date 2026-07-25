"""
Cardiology smoke extraction task.

Smoke-test scope (expand later to the full cardiology variable list):
  1. sternal_wound_infection  -> Keine | Oberflächlich | Tief
  2. reoperation_required     -> Nein | Ja | Unbekannt

Plus audit fields (German LLM output):
  - reasoning
  - evidence_quotes
  - information_sufficient

Input: patient-level Diagnoseliste built from HER_Diagnose_*.csv (Excel optional).
Prompts and model outputs are German; code identifiers stay English.
"""

from __future__ import annotations

from configs.tasks.base import EvidenceGroup, ExtractionTask, SchemaField

TASK = ExtractionTask(
    name="cardiology_smoke",
    description=(
        "Smoke-Test: Sternale Wundinfektion und Re-Operation erforderlich "
        "aus einer Patienten-Diagnoseliste extrahieren, mit deutscher Begründung."
    ),
    language="de",
    send_full_text_when_no_evidence=True,
    fields=(
        SchemaField(
            name="sternal_wound_infection",
            type="enum",
            enum=("Keine", "Oberflächlich", "Tief"),
            required=True,
            default="Keine",
            description="Schweregrad der sternalen Wundinfektion.",
        ),
        SchemaField(
            name="reoperation_required",
            type="enum",
            enum=("Nein", "Ja", "Unbekannt"),
            required=True,
            default="Unbekannt",
            description="Ob eine Re-Operation erforderlich war.",
        ),
        SchemaField(
            name="information_sufficient",
            type="boolean",
            required=True,
            default=False,
            description=(
                "True, wenn die Diagnoseliste genug Information enthält, um die "
                "klinischen Felder mit angemessener Sicherheit zu setzen."
            ),
        ),
        SchemaField(
            name="reasoning",
            type="string",
            required=True,
            default="",
            description="Schrittweise klinische Begründung auf Deutsch.",
        ),
        SchemaField(
            name="evidence_quotes",
            type="array",
            required=True,
            default=[],
            description="Wörtliche Zitate aus der Diagnoseliste als Belege.",
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
            "type": "normalize",
            "field": "sternal_wound_infection",
            "map": {
                "None": "Keine",
                "none": "Keine",
                "Superficial": "Oberflächlich",
                "superficial": "Oberflächlich",
                "Deep": "Tief",
                "deep": "Tief",
                "Oberflaechlich": "Oberflächlich",
            },
        },
        {
            "type": "normalize",
            "field": "reoperation_required",
            "map": {
                "No": "Nein",
                "no": "Nein",
                "Yes": "Ja",
                "yes": "Ja",
                "Unknown": "Unbekannt",
                "unknown": "Unbekannt",
            },
        },
        {
            "type": "requires",
            "if": {"information_sufficient": True},
            "then_required": ["reasoning", "sternal_wound_infection", "reoperation_required"],
        },
    ),
)
