"""
Cardiology smoke extraction task (Diagnoseliste).

Clinical variables (one LLM call each; one patient row):
  - new_permanent_pacemaker      Nein|Ja|Unbekannt
  - postop_atrial_fibrillation   Nein|Ja|Unbekannt
  - cerebrovascular_event        Keine|TIA|Schlaganfall|Unbekannt
  - sternal_wound_infection      Nein|Ja|Unbekannt
  - reoperation_required         Nein|Ja|Unbekannt
  - reoperation_context          Freitext (Kontext rund um Re-Operation)
  - multi_system_failure         Nein|Ja|Unbekannt
  - rethoracotomy                Nein|Ja|Unbekannt
  - rethoracotomy_context        Freitext (Kontext rund um Re-Thorakotomie)
  - liver_cirrhosis              Nein|Ja|Unbekannt

Fine severity (oberflächlich/tief, Child-Pugh, …) is deferred.
Keywords/prompts/outputs: German; field names: English.
"""

from __future__ import annotations

from typing import Dict, Tuple

from configs.tasks.base import (
    EvidenceGroup,
    ExtractionTask,
    SchemaField,
    VariableSpec,
)

_YN_ENUM = ("Nein", "Ja", "Unbekannt")

_YN_NORMALIZE: Dict[str, str] = {
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

_CVA_ENUM = ("Keine", "TIA", "Schlaganfall", "Unbekannt")
_CVA_NORMALIZE: Dict[str, str] = {
    "None": "Keine",
    "none": "Keine",
    "Nein": "Keine",
    "nein": "Keine",
    "No": "Keine",
    "no": "Keine",
    "TIA": "TIA",
    "tia": "TIA",
    "Transient ischemic attack": "TIA",
    "transient ischemic attack": "TIA",
    "Stroke": "Schlaganfall",
    "stroke": "Schlaganfall",
    "Hirnschlag": "Schlaganfall",
    "hirnschlag": "Schlaganfall",
    "Apoplex": "Schlaganfall",
    "apoplex": "Schlaganfall",
    "CVI": "Schlaganfall",
    "cvi": "Schlaganfall",
    "Unknown": "Unbekannt",
    "unknown": "Unbekannt",
    "Unbekannt": "Unbekannt",
    "unbekannt": "Unbekannt",
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


def _yn_field(name: str, description: str) -> SchemaField:
    return SchemaField(
        name=name,
        type="enum",
        enum=_YN_ENUM,
        required=True,
        default="Unbekannt",
        description=description,
    )


def _yn_variable(
    *,
    name: str,
    label: str,
    prompt_name: str,
    description: str,
    evidence_group_names: Tuple[str, ...],
) -> VariableSpec:
    field = _yn_field(name, description)
    return VariableSpec(
        name=name,
        label=label,
        prompt_name=prompt_name,
        fields=(field, *_AUDIT_FIELDS),
        evidence_group_names=evidence_group_names,
        consistency_rules=(
            {"type": "normalize", "field": name, "map": _YN_NORMALIZE},
            {
                "type": "requires",
                "if": {"information_sufficient": True},
                "then_required": ["reasoning", name],
            },
        ),
    )


def _text_variable(
    *,
    name: str,
    label: str,
    prompt_name: str,
    description: str,
    evidence_group_names: Tuple[str, ...],
) -> VariableSpec:
    field = SchemaField(
        name=name,
        type="string",
        required=True,
        default="",
        description=description,
    )
    return VariableSpec(
        name=name,
        label=label,
        prompt_name=prompt_name,
        fields=(field, *_AUDIT_FIELDS),
        evidence_group_names=evidence_group_names,
        consistency_rules=(
            {
                "type": "requires",
                "if": {"information_sufficient": True},
                "then_required": ["reasoning", name],
            },
        ),
    )


_FIELD_PACEMAKER = _yn_field(
    "new_permanent_pacemaker",
    "Neuer permanenter Schrittmacher im postoperativen Verlauf (Nein/Ja/Unbekannt).",
)
_FIELD_AF = _yn_field(
    "postop_atrial_fibrillation",
    "Neu aufgetretenes postoperatives Vorhofflimmern (Nein/Ja/Unbekannt).",
)
_FIELD_CVA = SchemaField(
    name="cerebrovascular_event",
    type="enum",
    enum=_CVA_ENUM,
    required=True,
    default="Keine",
    description=(
        "Neues zerebrovaskuläres Ereignis: Keine / TIA / Schlaganfall / Unbekannt. "
        "V.a./Verdacht allein zählt nicht als positiver Beleg → Unbekannt."
    ),
)
_FIELD_SWI = _yn_field(
    "sternal_wound_infection",
    "Sternale Wundinfektion dokumentiert (Nein/Ja/Unbekannt; ohne Schweregrad).",
)
_FIELD_REOP = _yn_field(
    "reoperation_required",
    "Re-Operation erforderlich/durchgeführt (Nein/Ja/Unbekannt).",
)
_FIELD_REOP_CTX = SchemaField(
    name="reoperation_context",
    type="string",
    required=True,
    default="",
    description="Relevanter Freitext/Kontext rund um Re-Operation (kein Enum).",
)
_FIELD_MSF = _yn_field(
    "multi_system_failure",
    "Multi-Organ-Versagen / Multi-system failure (Nein/Ja/Unbekannt).",
)
_FIELD_RETHOR = _yn_field(
    "rethoracotomy",
    "Re-Thorakotomie durchgeführt (Nein/Ja/Unbekannt).",
)
_FIELD_RETHOR_CTX = SchemaField(
    name="rethoracotomy_context",
    type="string",
    required=True,
    default="",
    description="Relevanter Freitext/Kontext rund um Re-Thorakotomie (kein Enum).",
)
_FIELD_CIRRHOSIS = _yn_field(
    "liver_cirrhosis",
    "Leberzirrhose dokumentiert (Nein/Ja/Unbekannt; ohne Child-Pugh). "
    "V.a./Verdacht allein zählt nicht als positiver Beleg → Unbekannt.",
)

_EVIDENCE_GROUPS = (
    EvidenceGroup(
        name="schrittmacher",
        role="positive",
        priority=1,
        phrases=(
            "schrittmacher",
            "herzschrittmacher",
            "sm-implantation",
            "sm implantation",
            "schrittmacherimplantation",
            "schrittmacher-implantation",
            "icd-implantation",
            "icd implantation",
            "crt-d",
            "crt-p",
            "aggregatwechsel",
            "permanenter schrittmacher",
            "neuer schrittmacher",
        ),
    ),
    EvidenceGroup(
        name="vorhofflimmern",
        role="positive",
        priority=1,
        phrases=(
            "vorhofflimmern",
            "vorhofflattern",
            "vhf",
            "postoperatives vorhofflimmern",
            "neues vorhofflimmern",
            "erstmaliges vorhofflimmern",
            "neu aufgetretenes vorhofflimmern",
        ),
    ),
    EvidenceGroup(
        name="cva",
        role="positive",
        priority=1,
        phrases=(
            "schlaganfall",
            "hirnschlag",
            "apoplex",
            "apoplexie",
            "tia",
            "transitorische ischämische attacke",
            "transitorische ischaemische attacke",
            "zerebrovaskulär",
            "cerebrovaskulär",
            "cvi",
            "insult",
            "hirninfarkt",
            "hirnblutung",
            "intrazerebrale blutung",
        ),
    ),
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
        name="multi_organ",
        role="positive",
        priority=1,
        phrases=(
            "multiorganversagen",
            "multi-organ-versagen",
            "mehrorganversagen",
            "mov",
            "mods",
            "systemisches organversagen",
            "organversagen",
            "septischer schock",
            "distributiver schock",
        ),
    ),
    EvidenceGroup(
        name="rethorakotomie",
        role="positive",
        priority=1,
        phrases=(
            "rethorakotomie",
            "re-thorakotomie",
            "re thoracotomie",
            "erneute thorakotomie",
            "secondlook thorakotomie",
            "revision thorakotomie",
        ),
    ),
    EvidenceGroup(
        name="leberzirrhose",
        role="positive",
        priority=1,
        phrases=(
            "leberzirrhose",
            "lebercirrhose",
            "zirrhose",
            "cirrhose",
            "child-pugh",
            "child pugh",
            "leberinsuffizienz",
            "chronische lebererkrankung",
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
            "kein schrittmacher",
            "kein vorhofflimmern",
            "kein schlaganfall",
            "keine tia",
            "keine zirrhose",
            "keine leberzirrhose",
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
    (r"\bkein(?:e|en)?\b.{0,40}?\bschrittmacher", "kein_schrittmacher"),
    (r"\bkein(?:e|en)?\b.{0,40}?\bvorhofflimmern", "kein_vhf"),
    (r"\bkein(?:e|en)?\b.{0,40}?\bschlaganfall", "kein_schlaganfall"),
    (r"\bkein(?:e|en)?\b.{0,40}?\bzirrhose", "keine_zirrhose"),
)

_SECTION_MARKERS = (("[Diagnoseliste]", "diagnoseliste"),)

VARIABLE_PACEMAKER = _yn_variable(
    name="new_permanent_pacemaker",
    label="Neuer permanenter Schrittmacher",
    prompt_name="cardiology_var_pacemaker",
    description=_FIELD_PACEMAKER.description,
    evidence_group_names=("schrittmacher", "verneinung"),
)
VARIABLE_AF = _yn_variable(
    name="postop_atrial_fibrillation",
    label="Postoperatives Vorhofflimmern",
    prompt_name="cardiology_var_af",
    description=_FIELD_AF.description,
    evidence_group_names=("vorhofflimmern", "verneinung"),
)
VARIABLE_CVA = VariableSpec(
    name="cerebrovascular_event",
    label="Zerebrovaskuläres Ereignis",
    prompt_name="cardiology_var_cva",
    fields=(_FIELD_CVA, *_AUDIT_FIELDS),
    evidence_group_names=("cva", "verneinung"),
    consistency_rules=(
        {"type": "normalize", "field": "cerebrovascular_event", "map": _CVA_NORMALIZE},
        {
            "type": "requires",
            "if": {"information_sufficient": True},
            "then_required": ["reasoning", "cerebrovascular_event"],
        },
    ),
)
VARIABLE_SWI = _yn_variable(
    name="sternal_wound_infection",
    label="Sternale Wundinfektion",
    prompt_name="cardiology_smoke_swi",
    description=_FIELD_SWI.description,
    evidence_group_names=("sternale_wundinfektion", "wund_kontext", "verneinung"),
)
VARIABLE_REOP = _yn_variable(
    name="reoperation_required",
    label="Re-Operation erforderlich",
    prompt_name="cardiology_smoke_reop",
    description=_FIELD_REOP.description,
    evidence_group_names=("reeingriff", "verneinung"),
)
VARIABLE_REOP_CTX = _text_variable(
    name="reoperation_context",
    label="Re-Operation Kontext",
    prompt_name="cardiology_var_reop_context",
    description=_FIELD_REOP_CTX.description,
    evidence_group_names=("reeingriff", "verneinung"),
)
VARIABLE_MSF = _yn_variable(
    name="multi_system_failure",
    label="Multi-Organ-Versagen",
    prompt_name="cardiology_var_msf",
    description=_FIELD_MSF.description,
    evidence_group_names=("multi_organ", "verneinung"),
)
VARIABLE_RETHOR = _yn_variable(
    name="rethoracotomy",
    label="Re-Thorakotomie",
    prompt_name="cardiology_var_rethoracotomy",
    description=_FIELD_RETHOR.description,
    evidence_group_names=("rethorakotomie", "reeingriff", "verneinung"),
)
VARIABLE_RETHOR_CTX = _text_variable(
    name="rethoracotomy_context",
    label="Re-Thorakotomie Kontext",
    prompt_name="cardiology_var_rethor_context",
    description=_FIELD_RETHOR_CTX.description,
    evidence_group_names=("rethorakotomie", "reeingriff", "verneinung"),
)
VARIABLE_CIRRHOSIS = _yn_variable(
    name="liver_cirrhosis",
    label="Leberzirrhose",
    prompt_name="cardiology_var_cirrhosis",
    description=_FIELD_CIRRHOSIS.description,
    evidence_group_names=("leberzirrhose", "verneinung"),
)

_CLINICAL_FIELDS = (
    _FIELD_PACEMAKER,
    _FIELD_AF,
    _FIELD_CVA,
    _FIELD_SWI,
    _FIELD_REOP,
    _FIELD_REOP_CTX,
    _FIELD_MSF,
    _FIELD_RETHOR,
    _FIELD_RETHOR_CTX,
    _FIELD_CIRRHOSIS,
)

_VARIABLES = (
    VARIABLE_PACEMAKER,
    VARIABLE_AF,
    VARIABLE_CVA,
    VARIABLE_SWI,
    VARIABLE_REOP,
    VARIABLE_REOP_CTX,
    VARIABLE_MSF,
    VARIABLE_RETHOR,
    VARIABLE_RETHOR_CTX,
    VARIABLE_CIRRHOSIS,
)

TASK = ExtractionTask(
    name="cardiology_smoke",
    description=(
        "Kardiologie Diagnoseliste: mehrere klinische Variablen, "
        "je eigene LLM-Abfrage; eine Ergebniszeile pro Patient."
    ),
    language="de",
    send_full_text_when_no_evidence=True,
    fields=(*_CLINICAL_FIELDS, *_AUDIT_FIELDS),
    evidence_groups=_EVIDENCE_GROUPS,
    section_markers=_SECTION_MARKERS,
    negation_patterns=_NEGATION_PATTERNS,
    prompt_name="",
    consistency_rules=(
        {"type": "normalize", "field": "new_permanent_pacemaker", "map": _YN_NORMALIZE},
        {"type": "normalize", "field": "postop_atrial_fibrillation", "map": _YN_NORMALIZE},
        {"type": "normalize", "field": "cerebrovascular_event", "map": _CVA_NORMALIZE},
        {"type": "normalize", "field": "sternal_wound_infection", "map": _YN_NORMALIZE},
        {"type": "normalize", "field": "reoperation_required", "map": _YN_NORMALIZE},
        {"type": "normalize", "field": "multi_system_failure", "map": _YN_NORMALIZE},
        {"type": "normalize", "field": "rethoracotomy", "map": _YN_NORMALIZE},
        {"type": "normalize", "field": "liver_cirrhosis", "map": _YN_NORMALIZE},
        {
            "type": "requires",
            "if": {"information_sufficient": True},
            "then_required": [
                "reasoning",
                "new_permanent_pacemaker",
                "postop_atrial_fibrillation",
                "cerebrovascular_event",
                "sternal_wound_infection",
                "reoperation_required",
                "multi_system_failure",
                "rethoracotomy",
                "liver_cirrhosis",
            ],
        },
    ),
    variables=_VARIABLES,
)
