"""
Format evidence quotes with source-section labels and field-local reasoning.

Used by Dendrite/Rodney review exports so reviewers see e.g.
  diagnose: "Schrittmacher neu implantiert"
and only the reasoning for the variable of that row.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.preprocessing.report_identity import normalize_str

FIELD_TEXT_SOURCE: Dict[str, str] = {
    "pacemaker": "verlegung",
    "atrial_fibrillation": "both",
    "cerebrovascular_event": "both",
    "reoperation_required": "both",
    "multi_system_failure": "verlegung",
}

# Canonical section marker → display name in review Excel
SECTION_DISPLAY: Dict[str, str] = {
    "diag": "diagnose",
    "diagnose": "diagnose",
    "Diagnose_Value": "diagnose",
    "diagnoseliste": "diagnose",
    "epikrise": "epikrise",
    "jetziges_leiden": "jetziges_leiden",
    "jetztleid": "jetziges_leiden",
    "prozedere": "prozedere",
    "procedere": "prozedere",
    "stat_ein": "stat_ein",
    "anamn": "anamn",
    "anamnese": "anamn",
    "Austrittsbericht": "austritt",
    "Verlegungsbericht": "verlegung",
    "Diagnoseliste": "diagnose",
}

# field → headings that may appear in aggregated ### reasoning blocks
FIELD_REASONING_HEADINGS: Dict[str, Tuple[str, ...]] = {
    "pacemaker": ("Schrittmacher", "pacemaker", "Permanenter Schrittmacher"),
    "atrial_fibrillation": ("Vorhofflimmern", "atrial_fibrillation", "AF"),
    "cerebrovascular_event": (
        "Zerebrovaskuläres Ereignis",
        "cerebrovascular_event",
        "CVA",
        "Schlaganfall",
    ),
    "reoperation_required": (
        "Re-Operation",
        "reoperation_required",
        "Re-Operation erforderlich",
        "erneute Operation",
    ),
    "reoperation_context": ("Re-Operation Kontext", "reoperation_context"),
    "multi_system_failure": (
        "Multi-Organ-Versagen",
        "multi_system_failure",
        "MOV",
    ),
    "rethoracotomy": ("Re-Thorakotomie", "rethoracotomy"),
    "rethoracotomy_context": ("Re-Thorakotomie Kontext", "rethoracotomy_context"),
    "liver_cirrhosis": ("Leberzirrhose", "liver_cirrhosis", "Zirrhose"),
}

_SECTION_SPLIT = re.compile(r"\[([^\]]+)\]\s*")
_HEADING_SPLIT = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


def _clean(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s


def parse_quotes_list(raw: Any) -> List[str]:
    s = _clean(raw)
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [_clean(x) for x in parsed if _clean(x)]
    except Exception:
        pass
    if " | " in s:
        return [_clean(x) for x in s.split(" | ") if _clean(x)]
    return [s]


def parse_sectioned_text(text: str) -> Dict[str, str]:
    """Parse ``[diag]\\n...\\n\\n[epikrise]\\n...`` style report text into sections."""
    text = _clean(text)
    if not text:
        return {}
    parts = _SECTION_SPLIT.split(text)
    # parts: [preamble, name1, body1, name2, body2, ...]
    out: Dict[str, str] = {}
    if len(parts) == 1:
        out["_full"] = parts[0]
        return out
    preamble = parts[0].strip()
    if preamble:
        out["_preamble"] = preamble
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        key = name
        out[key] = body
        display = SECTION_DISPLAY.get(name) or SECTION_DISPLAY.get(name.lower())
        if display:
            out.setdefault(display, body)
    return out


def _normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def find_section_for_quote(quote: str, sections: Dict[str, str]) -> str:
    q = _normalize_for_match(quote)
    if not q:
        return "unbekannt"
    # Prefer real sections over _full/_preamble
    candidates = [
        (k, v)
        for k, v in sections.items()
        if not k.startswith("_") and _clean(v)
    ]
    for key, body in candidates:
        if q in _normalize_for_match(body):
            return SECTION_DISPLAY.get(key) or SECTION_DISPLAY.get(key.lower()) or key
    # Short quotes: substring either way
    for key, body in candidates:
        bn = _normalize_for_match(body)
        if len(q) >= 12 and (q in bn or bn in q):
            return SECTION_DISPLAY.get(key) or SECTION_DISPLAY.get(key.lower()) or key
    if "_full" in sections and q in _normalize_for_match(sections["_full"]):
        return "text"
    return "unbekannt"


def format_tagged_evidence(
    quotes: Sequence[str],
    *,
    source_text: str = "",
    fallback_column: str = "",
) -> str:
    """
    Format quotes as ``diagnose: "..." | epikrise: "..."``.

    Uses section markers in *source_text* when available; otherwise
    *fallback_column* (first listed source column) as the label.
    """
    clean_quotes = [_clean(q) for q in quotes if _clean(q)]
    if not clean_quotes:
        return ""
    sections = parse_sectioned_text(source_text)
    fallback = _clean(fallback_column).split(",")[0].strip()
    fallback = SECTION_DISPLAY.get(fallback) or SECTION_DISPLAY.get(fallback.lower()) or fallback or "quelle"
    parts: List[str] = []
    for q in clean_quotes:
        if sections:
            label = find_section_for_quote(q, sections)
            if label == "unbekannt" and fallback:
                label = fallback
        else:
            label = fallback
        parts.append(f'{label}: "{q}"')
    return " | ".join(parts)


def extract_field_reasoning(aggregated: str, field: str) -> str:
    """Pull only the ``### <label>`` block for *field* from multi-variable reasoning."""
    text = _clean(aggregated)
    if not text:
        return ""
    headings = FIELD_REASONING_HEADINGS.get(field, (field,))
    matches = list(_HEADING_SPLIT.finditer(text))
    if not matches:
        # Single-variable blob without headings — only OK if short / no other ### 
        return ""
    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        title_l = title.lower()
        if not any(h.lower() in title_l or title_l in h.lower() for h in headings):
            continue
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        return text[start:end].strip().replace("\n", " ")
    return ""


def reasoning_for_field(pred: Dict[str, Any], field: str) -> str:
    """Prefer per-variable column; else parse ### block; never dump all variables."""
    specific = _clean(pred.get(f"{field}_reasoning"))
    if specific:
        # If somehow the whole multi-block landed in the per-field column, still trim.
        if "### " in specific and FIELD_REASONING_HEADINGS.get(field):
            trimmed = extract_field_reasoning(specific, field)
            return trimmed or specific.replace("\n", " ")
        return specific.replace("\n", " ")
    return extract_field_reasoning(_clean(pred.get("reasoning")), field)


def source_text_for_field(pred: Dict[str, Any], field: str, text_by_fall: Optional[Dict[str, str]] = None) -> str:
    """Best available source text for locating quote sections."""
    src = FIELD_TEXT_SOURCE.get(field, "report")
    verl = _clean(pred.get("verlegung_text"))
    diag = _clean(pred.get("diagnoseliste_text"))
    austr = _clean(pred.get("austritt_text"))
    fall = normalize_str(pred.get("verlegung_fallnr") or pred.get("report_id") or "")
    if text_by_fall and fall and fall in text_by_fall:
        loaded = text_by_fall[fall]
        if src == "verlegung":
            return loaded
        if src == "both":
            return "\n\n".join(p for p in (diag, loaded) if p) or loaded
        if src == "austritt":
            return austr or loaded
    if src == "verlegung":
        return verl
    if src == "diagnoseliste":
        return diag
    if src == "austritt":
        return austr
    if src == "both":
        return "\n\n".join(p for p in (diag, verl) if p)
    return verl or diag or _clean(pred.get("report_text"))


def load_verlegung_text_by_fall() -> Dict[str, str]:
    """Load IPS Verlegung texts keyed by FallNummer (best-effort; empty if no raw data)."""
    try:
        from src.preprocessing.verlegung_loader import (
            VERLEGUNG_TEXT_KEY,
            discover_her_verlegung_paths,
            load_verlegung_by_fall,
        )
    except Exception:
        return {}
    paths = discover_her_verlegung_paths()
    if not paths:
        return {}
    by_fall = load_verlegung_by_fall(paths)
    return {
        fall: normalize_str(info.get(VERLEGUNG_TEXT_KEY, ""))
        for fall, info in by_fall.items()
        if normalize_str(info.get(VERLEGUNG_TEXT_KEY, ""))
    }
