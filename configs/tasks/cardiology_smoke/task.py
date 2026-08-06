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
Rule keywords, prompts and model outputs are German; code field names stay English.
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
                "true, wenn die Diagnoseliste genug Information enthält, um die "
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
            name="sternale_wundinfektion",
            role="positive",
            priority=1,
            phrases=(
                "mediastinitis",
                "sternale wundinfektion",
                "sternale wundinfekt",
                "sternaler wundinfekt",
                "sternaler wundinfektion",
                "sternalem wundinfekt",
                "sternales wundinfekt",
                "sternumwundinfekt",
                "sternumwundinfektion",
                "wundinfekt sternum",
                "wundinfektion sternum",
                "wundinfekt sternal",
                "wundinfektion sternal",
                "sternale infektion",
                "sternale infekt",
                "tiefe sternale",
                "tiefer sternaler",
                "tief sternale",
                "oberflächliche sternale",
                "oberflaechliche sternale",
                "oberflächlicher sternaler",
                "osteomyelitis sternum",
                "sternumosteomyelitis",
                "sternalem vac",
                "sternales vac",
                "sternale vac",
                "vac sternum",
                "offener thorax",
                "offen belassener thorax",
            ),
        ),
        EvidenceGroup(
            name="reeingriff",
            role="positive",
            priority=1,
            phrases=(
                "re-operation",
                "reoperation",
                "reeingriff",
                "re-eingriff",
                "revisionseingriff",
                "revisionsoperation",
                "revisions-operation",
                "erneute operation",
                "nochmalige operation",
                "rethorakotomie",
                "re-thorakotomie",
                "resternotomie",
                "re-sternotomie",
                "nachblutung revision",
                "tamponade revision",
                "wundrevision",
                "wund-revision",
                "secondlook",
                "second-look",
                "second look",
                "verschluss thorakotomie",
                "thorakotomie",
                "débridement",
                "debridement",
                "débridément",
                "revision wegen",
                "chirurgische revision",
            ),
        ),
        EvidenceGroup(
            name="wund_kontext",
            role="context",
            priority=2,
            phrases=(
                "wundinfekt",
                "wundinfektion",
                "dehiszenz",
                "wunddehiszenz",
                "sternum",
                "sternal",
                "wunddébridement",
                "wunddebridement",
                "vac-therapie",
                "vac therapie",
                "vakuumtherapie",
            ),
        ),
        EvidenceGroup(
            name="verneinung",
            role="negation",
            priority=3,
            phrases=(
                "kein wundinfekt",
                "keine wundinfektion",
                "ohne wundinfekt",
                "kein reeingriff",
                "kein re-eingriff",
                "keine revision",
                "kein revisionseingriff",
                "ohne revision",
                "keine reoperation",
                "keine re-operation",
            ),
        ),
    ),
    section_markers=(
        ("[Diagnoseliste]", "diagnoseliste"),
    ),
    negation_patterns=(
        (r"\bkein(?:e|en)?\b.{0,30}?\bwundinfekt", "kein_wundinfekt"),
        (r"\bohne\b.{0,30}?\bwundinfekt", "ohne_wundinfekt"),
        (r"\bkein(?:e|en)?\b.{0,30}?\brevision", "keine_revision"),
        (r"\bkein(?:e|en)?\b.{0,30}?\bre-?operation", "keine_reoperation"),
        (r"\bkein(?:e|en)?\b.{0,30}?\breeingriff", "kein_reeingriff"),
        (r"\bohne\b.{0,30}?\brevision", "ohne_revision"),
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
