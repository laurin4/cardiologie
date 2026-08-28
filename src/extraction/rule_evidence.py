"""
Generic rule-based evidence extraction.

Scans a full report for the keyword groups declared by an extraction task and
produces a bounded, section-labelled *evidence bundle* for the LLM (rather than
the full report). The matching engine is entirely task-agnostic: keyword groups,
section markers, and negation patterns all come from the task specification.

Returns an evidence dict consumed by the LLM extraction stage and written to the
results CSV. If nothing actionable is found the LLM can be skipped.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from configs.tasks.base import ROLE_NEGATION, EvidenceGroup, ExtractionTask
from src.preprocessing.section_splitter import build_section_ranges, section_for_index

METHOD_NO_EVIDENCE = "no_evidence"
METHOD_STRUCTURED = "structured_evidence"

_ROLE_ORDER = {"positive": 1, "context": 2, "negation": 3}


# --------------------------------------------------------------------------- #
# Environment-tunable caps
# --------------------------------------------------------------------------- #
def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(minimum, v)


def _window_sentences() -> int:
    raw = os.environ.get("EVIDENCE_WINDOW_SENTENCES", "").strip()
    if not raw:
        return 2  # keyword ± 2 sentences so exclusions in the same/next sentence stay visible
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def _max_snippet_chars() -> int:
    return _int_env("EVIDENCE_MAX_SNIPPET_CHARS", 600, minimum=80)


def _max_snippets() -> int:
    return _int_env("EVIDENCE_MAX_SNIPPETS", 12, minimum=1)


def _max_llm_chars() -> int:
    return _int_env("EVIDENCE_MAX_LLM_CHARS", 8000, minimum=500)


def _max_hits_per_keyword() -> int:
    return _int_env("EVIDENCE_MAX_HITS_PER_KEYWORD", 3, minimum=1)


# --------------------------------------------------------------------------- #
# Normalized matching view (maps normalized index -> original char index)
# --------------------------------------------------------------------------- #
_LETTER_EXTRA = "äöüßÄÖÜ"


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch in _LETTER_EXTRA


def _build_matching_view(text: str) -> Tuple[str, List[int]]:
    """Lowercased, punctuation-as-space view plus a norm->orig index map."""
    if not text:
        return "", []
    norm_chars: List[str] = []
    norm_to_orig: List[int] = []
    i = 0
    n = len(text)
    need_space = False
    while i < n:
        ch = text[i]
        if _is_word_char(ch):
            if need_space and norm_chars and norm_chars[-1] != " ":
                norm_chars.append(" ")
                norm_to_orig.append(i)
            word_start = i
            letters: List[str] = []
            while i < n and _is_word_char(text[i]):
                letters.append(text[i].lower())
                i += 1
            for j, letter in enumerate(letters):
                norm_chars.append(letter)
                norm_to_orig.append(word_start + min(j, len(letters) - 1))
            need_space = False
        else:
            need_space = bool(norm_chars) and norm_chars[-1] != " "
            i += 1
    return "".join(norm_chars), norm_to_orig


def _norm_match_to_original_span(
    text: str, norm_start: int, norm_end: int, norm_to_orig: List[int]
) -> Tuple[int, int]:
    if norm_start >= norm_end or not norm_to_orig:
        return 0, 0
    orig_start = norm_to_orig[norm_start]
    orig_end = norm_to_orig[norm_end - 1] + 1
    while orig_start > 0 and _is_word_char(text[orig_start - 1]):
        orig_start -= 1
    while orig_end < len(text) and _is_word_char(text[orig_end]):
        orig_end += 1
    return orig_start, orig_end


def _phrase_matching_form(phrase: str) -> str:
    view, _ = _build_matching_view(phrase)
    return view


def _find_phrase_in_norm(norm_text: str, phrase: str, start: int) -> int:
    """
    Find *phrase* in *norm_text* from *start*.

    Single-token phrases require word boundaries (space or string edge) to avoid
    matching inside longer words; multi-token phrases already carry internal
    spaces and use plain substring search.
    """
    if " " in phrase:
        return norm_text.find(phrase, start)
    pos = start
    plen = len(phrase)
    while True:
        i = norm_text.find(phrase, pos)
        if i < 0:
            return -1
        before_ok = i == 0 or norm_text[i - 1] == " "
        after_idx = i + plen
        after_ok = after_idx >= len(norm_text) or norm_text[after_idx] == " "
        if before_ok and after_ok:
            return i
        pos = i + 1


# --------------------------------------------------------------------------- #
# Sentence windows
# --------------------------------------------------------------------------- #
def _split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for p in parts:
        if not p:
            continue
        start = text.find(p, pos)
        if start < 0:
            start = pos
        end = start + len(p)
        spans.append((start, end))
        pos = end
    return spans or [(0, len(text))]


def _sentence_idx_for_pos(sentence_spans: List[Tuple[int, int]], pos: int) -> int:
    for i, (a, b) in enumerate(sentence_spans):
        if a <= pos < b:
            return i
    return 0


def _window_text(
    text: str,
    sentence_spans: List[Tuple[int, int]],
    match_start: int,
    window_sentences: int,
    max_chars: int,
) -> str:
    si = _sentence_idx_for_pos(sentence_spans, match_start)
    lo = max(0, si - window_sentences)
    hi = min(len(sentence_spans) - 1, si + window_sentences)
    a = sentence_spans[lo][0]
    b = sentence_spans[hi][1]
    chunk = re.sub(r"\s+", " ", text[a:b]).strip()
    if len(chunk) > max_chars:
        chunk = chunk[: max_chars - 1] + "\u2026"
    return chunk


# --------------------------------------------------------------------------- #
# Task-driven keyword flattening + negation
# --------------------------------------------------------------------------- #
def _flatten_keywords(
    groups: Tuple[EvidenceGroup, ...]
) -> List[Tuple[str, str, str]]:
    """Return ``(normalized_phrase, group_name, role)`` sorted by length desc."""
    out: List[Tuple[str, str, str]] = []
    for group in groups:
        for phrase in group.phrases:
            norm = _phrase_matching_form(phrase)
            if norm:
                out.append((norm, group.name, group.role))
    out.sort(key=lambda x: len(x[0]), reverse=True)
    return out


def _span_overlaps_used(used: List[Tuple[int, int]], start: int, end: int) -> bool:
    for u, v in used:
        if not (end <= u or start >= v):
            return True
    return False


def _apply_negation_patterns(
    src: str,
    patterns: Tuple[Tuple[str, str], ...],
    raw_matches: List[Tuple[int, int, str, str, str]],
    used: List[Tuple[int, int]],
) -> None:
    """Apply sentence-local negation regexes before keyword matching."""
    if not patterns:
        return
    compiled = [(re.compile(rx, re.DOTALL), label) for rx, label in patterns]
    for sent_start, sent_end in _split_sentence_spans(src):
        sent = src[sent_start:sent_end]
        sent_norm, sent_map = _build_matching_view(sent)
        if not sent_norm.strip():
            continue
        for pattern, label in compiled:
            for match in pattern.finditer(sent_norm):
                rel_start, rel_end = _norm_match_to_original_span(
                    sent, match.start(), match.end(), sent_map
                )
                orig_start = sent_start + rel_start
                orig_end = sent_start + rel_end
                if _span_overlaps_used(used, orig_start, orig_end):
                    continue
                raw_matches.append((orig_start, orig_end, label, "__negation__", ROLE_NEGATION))
                used.append((orig_start, orig_end))


# --------------------------------------------------------------------------- #
# Snippet post-processing
# --------------------------------------------------------------------------- #
def _cap_raw_matches(
    matches: List[Tuple[int, int, str, str, str]]
) -> List[Tuple[int, int, str, str, str]]:
    counts: Dict[Tuple[str, str], int] = {}
    cap = _max_hits_per_keyword()
    out: List[Tuple[int, int, str, str, str]] = []
    for item in matches:
        keyword, group = item[2], item[3]
        key = (group, keyword.lower())
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > cap:
            continue
        out.append(item)
    return out


def _dedupe_snippets(snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for s in snippets:
        text_norm = re.sub(r"\s+", " ", (s.get("text") or "").strip().lower())[:200]
        key = (s.get("section"), s.get("evidence_group"), s.get("keyword"), text_norm)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _sort_snippets(
    snippets: List[Dict[str, Any]], section_priority: Dict[str, int]
) -> List[Dict[str, Any]]:
    def key(s: Dict[str, Any]) -> Tuple[int, int, str]:
        sp = section_priority.get(str(s.get("section") or "unknown"), 99)
        ro = _ROLE_ORDER.get(str(s.get("role") or ""), 99)
        return (sp, ro, str(s.get("text") or ""))

    return sorted(snippets, key=key)


_ROLE_LABEL_DE = {
    "positive": "positiv",
    "context": "kontext",
    "negation": "verneinung",
}


def _snippet_label_line(snippet: Dict[str, Any], *, language: str) -> str:
    role = str(snippet.get("role") or "")
    if language.startswith("de"):
        role_label = _ROLE_LABEL_DE.get(role, role)
        return (
            f"[Abschnitt={snippet.get('section')} | Gruppe={snippet.get('evidence_group')} | "
            f"Rolle={role_label} | Priorität={snippet.get('priority')}]"
        )
    return (
        f"[section={snippet.get('section')} | group={snippet.get('evidence_group')} | "
        f"role={role} | priority={snippet.get('priority')}]"
    )


def _build_llm_report_text(
    snippets: List[Dict[str, Any]],
    max_total_chars: int,
    *,
    language: str = "en",
) -> str:
    lang = (language or "en").lower()
    header = "Bericht-Evidenz-Snippets:" if lang.startswith("de") else "Report evidence snippets:"
    lines = [header, ""]
    for s in snippets:
        lines.append(_snippet_label_line(s, language=lang))
        lines.append(str(s.get("text") or ""))
        lines.append("")
    body = "\n".join(lines)
    if len(body) <= max_total_chars:
        return body
    keep = [header, ""]
    for s in snippets:
        block = (
            f"{_snippet_label_line(s, language=lang)}\n"
            f"{str(s.get('text') or '')}\n"
        )
        if sum(len(x) + 1 for x in keep) + len(block) > max_total_chars:
            break
        keep.append(block)
    return "\n".join(keep)


def snippets_json_for_csv(snippets: List[Dict[str, Any]]) -> str:
    """Stable JSON string for the results CSV column."""
    return json.dumps(snippets or [], ensure_ascii=False)


def llm_should_receive_evidence(snippets: List[Dict[str, Any]]) -> bool:
    """True when at least one non-negation snippet is present."""
    return any(str(s.get("role")) != ROLE_NEGATION for s in snippets)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _empty_bundle(original_len: int, group_names: List[str]) -> Dict[str, Any]:
    return {
        "original_report_text_length": original_len,
        "llm_report_text": "",
        "llm_report_text_length": 0,
        "reduction_method": METHOD_NO_EVIDENCE,
        "evidence_snippets": [],
        "keyword_hits_count": 0,
        "evidence_flags": {name: False for name in group_names},
        "has_positive_evidence": False,
        "has_negation": False,
    }


def extract_rule_evidence(report_text: str, task: ExtractionTask) -> Dict[str, Any]:
    """
    Scan *report_text* using *task* keyword groups; return an evidence bundle.

    The bundle contains section-labelled snippets, a bounded ``llm_report_text``
    to send to the LLM, per-group presence flags, and summary counts.
    """
    src = str(report_text or "")
    original_len = len(src)
    group_names = [g.name for g in task.evidence_groups]

    if not src.strip() or not task.evidence_groups:
        return _empty_bundle(original_len, group_names)

    norm_text, norm_to_orig = _build_matching_view(src)
    section_ranges = build_section_ranges(src, task.section_markers)
    section_priority = {label: i for i, (_, label) in enumerate(task.section_markers, start=1)}
    section_priority.setdefault("unknown", 99)
    keywords = _flatten_keywords(task.evidence_groups)
    role_by_group = {g.name: g.role for g in task.evidence_groups}

    raw_matches: List[Tuple[int, int, str, str, str]] = []
    used: List[Tuple[int, int]] = []

    _apply_negation_patterns(src, task.negation_patterns, raw_matches, used)

    for phrase, group_name, role in keywords:
        start = 0
        plen = len(phrase)
        while True:
            i = _find_phrase_in_norm(norm_text, phrase, start)
            if i < 0:
                break
            orig_start, orig_end = _norm_match_to_original_span(
                src, i, i + plen, norm_to_orig
            )
            if not _span_overlaps_used(used, orig_start, orig_end):
                raw_matches.append((orig_start, orig_end, phrase, group_name, role))
                used.append((orig_start, orig_end))
            start = i + max(1, plen)

    raw_matches.sort(key=lambda x: x[0])
    raw_matches = _cap_raw_matches(raw_matches)
    hit_count = len(raw_matches)

    sentence_spans = _split_sentence_spans(src)
    max_snip_chars = _max_snippet_chars()
    win = _window_sentences()

    snippets: List[Dict[str, Any]] = []
    for start, end, keyword, group_name, role in raw_matches:
        section = section_for_index(section_ranges, start)
        if not sentence_spans:
            tw = re.sub(r"\s+", " ", src[start:end]).strip()
            if len(tw) > max_snip_chars:
                tw = tw[: max_snip_chars - 1] + "\u2026"
        else:
            tw = _window_text(src, sentence_spans, start, win, max_snip_chars)
        snippets.append(
            {
                "section": section,
                "keyword": keyword,
                "evidence_group": group_name,
                "role": role,
                "priority": 0,
                "text": tw,
            }
        )

    snippets = _dedupe_snippets(snippets)
    snippets = _sort_snippets(snippets, section_priority)[: _max_snippets()]
    for idx, s in enumerate(snippets, start=1):
        s["priority"] = idx

    evidence_flags = {name: False for name in group_names}
    for s in snippets:
        evidence_flags[str(s.get("evidence_group"))] = True
    # Pattern-based negations use group "__negation__" (not a task EvidenceGroup).
    has_negation = any(str(s.get("role")) == ROLE_NEGATION for s in snippets)
    has_positive = llm_should_receive_evidence(snippets)

    method = METHOD_STRUCTURED if has_positive else METHOD_NO_EVIDENCE
    llm_body = (
        _build_llm_report_text(
            snippets,
            _max_llm_chars(),
            language=task.language or "en",
        )
        if has_positive
        else ""
    )

    return {
        "original_report_text_length": original_len,
        "llm_report_text": llm_body,
        "llm_report_text_length": len(llm_body),
        "reduction_method": method,
        "evidence_snippets": snippets,
        "keyword_hits_count": hit_count,
        "evidence_flags": evidence_flags,
        "has_positive_evidence": has_positive,
        "has_negation": has_negation,
    }
