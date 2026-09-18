"""
Structured evidence quotes: column + verbatim sentence for Rodney review.

Canonical export form:
  diagnose: "Schrittmacher neu implantiert" | epikrise: "paced AAI 90/min"
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
    "rethoracotomy": "verlegung",
    "rethoracotomy_context": "verlegung",
    "liver_cirrhosis": "austritt",
}

# Input aliases → display label used in Excel (exact reviewer-facing names)
SECTION_DISPLAY: Dict[str, str] = {
    "diag": "diag",
    "diagnose": "diag",
    "Diagnose_Value": "Diagnose_Value",
    "diagnose_value": "Diagnose_Value",
    "diagnoseliste": "Diagnose_Value",
    "epikrise": "epikrise",
    "jetziges_leiden": "jetziges_leiden",
    "jetztleid": "jetziges_leiden",
    "jetziges leiden": "jetziges_leiden",
    "prozedere": "prozedere",
    "procedere": "prozedere",
    "stat_ein": "stat_ein",
    "anamn": "anamn",
    "anamnese": "anamn",
    "Austrittsbericht": "austritt",
    "austritt": "austritt",
    "Verlegungsbericht": "verlegung",
    "verlegung": "verlegung",
    "Diagnoseliste": "Diagnose_Value",
}

ALLOWED_EVIDENCE_COLUMNS = (
    "diag",
    "diagnose",
    "Diagnose_Value",
    "epikrise",
    "jetziges_leiden",
    "jetztleid",
    "prozedere",
    "procedere",
    "stat_ein",
    "anamn",
    "anamnese",
)

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

EVIDENCE_SCHEMA_HINT_DE = (
    'Array von Objekten {"column":"<Spalte>","quote":"<wörtlicher Satz>"}; '
    "column eines von: diag, epikrise, jetziges_leiden, prozedere, Diagnose_Value, "
    "stat_ein, anamn; mind. 1 Zitat wenn Pred≠k.A. (alle relevanten Sätze, kein Max-3), sonst []"
)

_SECTION_SPLIT = re.compile(r"\[([^\]]+)\]\s*")
_HEADING_SPLIT = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_TAGGED_QUOTE = re.compile(
    r'^\s*([A-Za-zÄÖÜäöü_]+)\s*:\s*[«"„]?(.+?)[»"“]?\s*$', re.DOTALL
)


def _clean(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s


def display_column(raw_column: str) -> str:
    key = _clean(raw_column)
    if not key:
        return "unbekannt"
    return SECTION_DISPLAY.get(key) or SECTION_DISPLAY.get(key.lower()) or key.lower()


def normalize_evidence_quotes(raw: Any) -> List[Dict[str, str]]:
    """
    Coerce LLM evidence_quotes into ``[{"column": "...", "quote": "..."}, ...]``.

    Accepts:
    - list of dicts with column/quote (or source/text aliases)
    - list of plain strings (column=unbekannt)
    - list of ``diag: "..."`` strings
    - JSON string of any of the above
    """
    if raw is None:
        return []
    value = raw
    if isinstance(raw, str):
        s = _clean(raw)
        if not s:
            return []
        try:
            value = json.loads(s)
        except Exception:
            value = [s]

    if not isinstance(value, list):
        value = [value]

    out: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            col = _clean(
                item.get("column")
                or item.get("source")
                or item.get("section")
                or item.get("spalte")
            )
            quote = _clean(
                item.get("quote")
                or item.get("text")
                or item.get("zitat")
                or item.get("evidence")
            )
            if quote:
                out.append({"column": col or "unbekannt", "quote": quote})
            continue
        s = _clean(item)
        if not s:
            continue
        m = _TAGGED_QUOTE.match(s)
        if m:
            out.append({"column": m.group(1).strip(), "quote": m.group(2).strip()})
        else:
            out.append({"column": "unbekannt", "quote": s})
    return out


def format_structured_evidence(items: Sequence[Dict[str, str]]) -> str:
    """
    Reviewer format (exactly one source label per sentence):

      "jetziges_leiden": <Satz>. "diag": <Satz>. "epikrise": <Satz>.
    """
    parts: List[str] = []
    for item in items:
        quote = _clean(item.get("quote"))
        if not quote:
            continue
        label = display_column(item.get("column") or "unbekannt")
        # Strip wrapping quotes from the sentence if the model already added them.
        if (quote.startswith('"') and quote.endswith('"')) or (
            quote.startswith("«") and quote.endswith("»")
        ):
            quote = quote[1:-1].strip()
        parts.append(f'"{label}": {quote}')
    return " ".join(parts)


def columns_used_from_evidence(items: Sequence[Dict[str, str]]) -> str:
    """Unique display columns actually cited, semicolon-separated."""
    seen: List[str] = []
    for item in items:
        if not _clean(item.get("quote")):
            continue
        label = display_column(item.get("column") or "unbekannt")
        if label not in seen:
            seen.append(label)
    return "; ".join(seen)


def parse_quotes_list(raw: Any) -> List[str]:
    """Legacy: flat quote strings (for section-locator fallback)."""
    return [_clean(x.get("quote")) for x in normalize_evidence_quotes(raw) if _clean(x.get("quote"))]


def parse_sectioned_text(text: str) -> Dict[str, str]:
    """Parse ``[diag]\\n...\\n\\n[epikrise]\\n...`` style report text into sections."""
    text = _clean(text)
    if not text:
        return {}
    parts = _SECTION_SPLIT.split(text)
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
        out[name] = body
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
    candidates = [
        (k, v) for k, v in sections.items() if not k.startswith("_") and _clean(v)
    ]
    for key, body in candidates:
        if q in _normalize_for_match(body):
            return display_column(key)
    for key, body in candidates:
        bn = _normalize_for_match(body)
        if len(q) >= 12 and (q in bn or bn in q):
            return display_column(key)
    if "_full" in sections and q in _normalize_for_match(sections["_full"]):
        return "text"
    return "unbekannt"


def enrich_evidence_columns_from_text(
    items: Sequence[Dict[str, str]], source_text: str
) -> List[Dict[str, str]]:
    """If column is unbekannt, try to locate quote in sectioned source text."""
    sections = parse_sectioned_text(source_text)
    if not sections:
        return [dict(x) for x in items]
    out: List[Dict[str, str]] = []
    for item in items:
        col = _clean(item.get("column"))
        quote = _clean(item.get("quote"))
        if not quote:
            continue
        if not col or col.lower() in ("unbekannt", "unknown", "quelle", "text"):
            col = find_section_for_quote(quote, sections)
        out.append({"column": col, "quote": quote})
    return out


def format_tagged_evidence(
    quotes: Sequence[Any],
    *,
    source_text: str = "",
    fallback_column: str = "",
) -> str:
    """
    Format quotes for review Excel.

    *quotes* may be structured dicts or plain strings.
    """
    if quotes and isinstance(quotes[0], dict):
        items = [dict(x) for x in quotes]  # type: ignore[arg-type]
    else:
        items = [{"column": "", "quote": _clean(q)} for q in quotes if _clean(q)]
    if not items:
        return ""
    items = enrich_evidence_columns_from_text(items, source_text)
    for item in items:
        if display_column(item.get("column") or "") in ("unbekannt",) and fallback_column:
            item["column"] = fallback_column.split(",")[0].strip()
    return format_structured_evidence(items)


def extract_field_reasoning(aggregated: str, field: str) -> str:
    """Pull only the ``### <label>`` block for *field* from multi-variable reasoning."""
    text = _clean(aggregated)
    if not text:
        return ""
    headings = FIELD_REASONING_HEADINGS.get(field, (field,))
    matches = list(_HEADING_SPLIT.finditer(text))
    if not matches:
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
        if "### " in specific and FIELD_REASONING_HEADINGS.get(field):
            trimmed = extract_field_reasoning(specific, field)
            return trimmed or specific.replace("\n", " ")
        return specific.replace("\n", " ")
    return extract_field_reasoning(_clean(pred.get("reasoning")), field)


def source_text_for_field(
    pred: Dict[str, Any], field: str, text_by_fall: Optional[Dict[str, str]] = None
) -> str:
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
    """Load IPS Verlegung texts keyed by FallNummer (best-effort)."""
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


def backfill_evidence_from_rules(field: str, source_text: str) -> List[Dict[str, str]]:
    """
    When the LLM left evidence_quotes empty, rebuild cited sentences from the
    source text using the same keyword/section rules as the pipeline.
    """
    text = _clean(source_text)
    if not text or not field:
        return []
    try:
        from configs.tasks import load_task
        from src.extraction.rule_evidence import extract_rule_evidence
        from src.pipeline.pipeline import _variable_as_task
    except Exception:
        return []

    task = load_task("cardiology_smoke")
    var = next((v for v in (task.variables or ()) if v.name == field), None)
    if var is None:
        return []
    mini = _variable_as_task(task, var)
    bundle = extract_rule_evidence(text, mini)
    items: List[Dict[str, str]] = []
    seen: set[str] = set()
    for snip in bundle.get("evidence_snippets") or []:
        quote = _clean(snip.get("text"))
        if not quote:
            continue
        key = _normalize_for_match(quote)
        if key in seen:
            continue
        seen.add(key)
        section = _clean(snip.get("section")) or "unbekannt"
        if section.lower() in ("unknown", "unk", ""):
            section = "unbekannt"
        items.append({"column": section, "quote": quote})
    return items


def resolve_evidence_items(
    pred: Dict[str, Any],
    field: str,
    *,
    text_by_fall: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """LLM quotes first; otherwise keyword backfill from source text."""
    items = normalize_evidence_quotes(pred.get(f"{field}_evidence_quotes"))
    source_text = source_text_for_field(pred, field, text_by_fall=text_by_fall)
    if items:
        return enrich_evidence_columns_from_text(items, source_text)
    return enrich_evidence_columns_from_text(
        backfill_evidence_from_rules(field, source_text),
        source_text,
    )


def clinical_value_requires_evidence(fields: Dict[str, Any], task_fields: Sequence[Any]) -> bool:
    """True if the primary clinical pred is set and not k.A. (and schema has evidence_quotes)."""
    names = {getattr(f, "name", "") for f in task_fields}
    if "evidence_quotes" not in names:
        return False
    for f in task_fields:
        name = getattr(f, "name", "")
        if name in ("reasoning", "evidence_quotes", "information_sufficient"):
            continue
        val = _clean(fields.get(name))
        if not val:
            continue
        if val.lower() in ("k.a.", "k.a", "ka"):
            return False
        return True
    return False


def evidence_requirement_errors(fields: Dict[str, Any], task_fields: Sequence[Any]) -> List[str]:
    """Schema-style errors when quotes missing for a non-k.A. prediction."""
    if not clinical_value_requires_evidence(fields, task_fields):
        return []
    quotes = normalize_evidence_quotes(fields.get("evidence_quotes"))
    if quotes:
        return []
    return [
        "evidence_quotes must contain at least one {column, quote} object when prediction is not k.A."
    ]
