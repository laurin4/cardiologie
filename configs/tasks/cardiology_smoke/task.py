"""
Cardiology smoke extraction task.

Smoke-test scope (expand later to the full cardiology variable list):
  1. sternal_wound_infection  -> Nein | Ja | Unbekannt
  2. reoperation_required     -> Nein | Ja | Unbekannt

Each variable has its own LLM prompt + keyword subset (one call per variable).
Results are merged into one patient row. Fine-grained severity (oberflächlich/tief)
is deferred until clinical criteria / validation data exist.

Input: patient-level Diagnoseliste from HER_Diagnose_*.csv.
Rule keywords, prompts and model outputs are German; code field names stay English.
"""

from __future__ import annotations

from configs.tasks.base import (
    EvidenceGroup,
    ExtractionTask,
    SchemaField,
    VariableSpec,
)

_YN_ENUM = ("Nein", "Ja", "Unbekannt")

_YN_NORMALIZE = {
    "No": "Nein",
    "no": "Nein",
    "Yes": "Ja",
    "yes": "Ja",
    "Unknown": "Unbekannt",
    "unknown": "Unbekannt",
    "None": "Nein",
    "none": "Nein",
    "Keine": "Nein",
    "keine": "Nein",
    "Superficial": "Ja",
    "superficial": "Ja",
    "Oberflächlich": "Ja",
    "oberflächlich": "Ja",
    "Oberflaechlich": "Ja",
    "Deep": "Ja",
    "deep": "Ja",
    "Tief": "Ja",
    "tief": "Ja",
}

_AUDIT_FIELDS = (
    SchemaField(
        name="information_sufficient",
        type="boolean",
        required=True,
        default=False,
        description=(
            "true, wenn die Diagnoseliste genug Information enthält, um dieses "
            "Feld mit angemessener Sicherheit zu setzen."
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
)

_SWI_FIELD = SchemaField(
    name="sternal_wound_infection",
    type="enum",
    enum=_YN_ENUM,
    required=True,
    default="Unbekannt",
    description=(
        "Ob eine sternale Wundinfektion dokumentiert ist "
        "(Nein / Ja / Unbekannt; ohne Schweregrad)."
    ),
)

_REOP_FIELD = SchemaField(
    name="reoperation_required",
    type="enum",
    enum=_YN_ENUM,
    required=True,
    default="Unbekannt",
    description="Ob eine Re-Operation erforderlich war (Nein / Ja / Unbekannt).",
)

_EVIDENCE_GROUPS = (
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
)

_NEGATION_PATTERNS = (
    (r"\bkein(?:e|en)?\b.{0,30}?\bwundinfekt", "kein_wundinfekt"),
    (r"\bohne\b.{0,30}?\bwundinfekt", "ohne_wundinfekt"),
    (r"\bkein(?:e|en)?\b.{0,30}?\brevision", "keine_revision"),
    (r"\bkein(?:e|en)?\b.{0,30}?\bre-?operation", "keine_reoperation"),
    (r"\bkein(?:e|en)?\b.{0,30}?\breeingriff", "kein_reeingriff"),
    (r"\bohne\b.{0,30}?\brevision", "ohne_revision"),
)

_SECTION_MARKERS = (("[Diagnoseliste]", "diagnoseliste"),)

VARIABLE_SWI = VariableSpec(
    name="sternal_wound_infection",
    label="Sternale Wundinfektion",
    prompt_name="cardiology_smoke_swi",
    fields=(_SWI_FIELD, *_AUDIT_FIELDS),
    evidence_group_names=("sternale_wundinfektion", "wund_kontext", "verneinung"),
    consistency_rules=(
        {
            "type": "normalize",
            "field": "sternal_wound_infection",
            "map": _YN_NORMALIZE,
        },
        {
            "type": "requires",
            "if": {"information_sufficient": True},
            "then_required": ["reasoning", "sternal_wound_infection"],
        },
    ),
)

VARIABLE_REOP = VariableSpec(
    name="reoperation_required",
    label="Re-Operation erforderlich",
    prompt_name="cardiology_smoke_reop",
    fields=(_REOP_FIELD, *_AUDIT_FIELDS),
    evidence_group_names=("reeingriff", "verneinung"),
    consistency_rules=(
        {
            "type": "normalize",
            "field": "reoperation_required",
            "map": _YN_NORMALIZE,
        },
        {
            "type": "requires",
            "if": {"information_sufficient": True},
            "then_required": ["reasoning", "reoperation_required"],
        },
    ),
)

TASK = ExtractionTask(
    name="cardiology_smoke",
    description=(
        "Smoke-Test: Sternale Wundinfektion und Re-Operation erforderlich "
        "(je Variable eigene LLM-Abfrage; Ergebniszeile pro Patient)."
    ),
    language="de",
    send_full_text_when_no_evidence=True,
    fields=(_SWI_FIELD, _REOP_FIELD, *_AUDIT_FIELDS),
    evidence_groups=_EVIDENCE_GROUPS,
    section_markers=_SECTION_MARKERS,
    negation_patterns=_NEGATION_PATTERNS,
    prompt_name="",  # per-variable prompts via VariableSpec
    consistency_rules=(
        {
            "type": "normalize",
            "field": "sternal_wound_infection",
            "map": _YN_NORMALIZE,
        },
        {
            "type": "normalize",
            "field": "reoperation_required",
            "map": _YN_NORMALIZE,
        },
        {
            "type": "requires",
            "if": {"information_sufficient": True},
            "then_required": ["reasoning", "sternal_wound_infection", "reoperation_required"],
        },
    ),
    variables=(VARIABLE_SWI, VARIABLE_REOP),
)
