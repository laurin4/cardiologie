"""
Budget long report / Diagnoseliste text for LLM calls.

Only activates when the cleaned text exceeds a character budget. Short texts are
returned unchanged. Oversized texts are reduced to:
  - a short notice that truncation was applied
  - entry / section headers (so structure remains visible)
  - character windows around task keyword hits
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from configs.tasks.base import ExtractionTask


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def max_full_text_chars() -> int:
    """Hard budget for full-text fallback (override with EVIDENCE_MAX_FULL_TEXT_CHARS)."""
    return _int_env("EVIDENCE_MAX_FULL_TEXT_CHARS", 12000, minimum=2000)


def _window_chars() -> int:
    return _int_env("EVIDENCE_FULL_TEXT_WINDOW_CHARS", 500, minimum=120)


def _collect_phrases(task: ExtractionTask) -> List[str]:
    phrases: List[str] = []
    for group in task.evidence_groups:
        for phrase in group.phrases:
            p = str(phrase).strip()
            if p:
                phrases.append(p)
    # Longer phrases first so multi-word matches win.
    phrases.sort(key=len, reverse=True)
    return phrases


def _find_hit_spans(text: str, phrases: List[str]) -> List[Tuple[int, int, str]]:
    lower = text.lower()
    hits: List[Tuple[int, int, str]] = []
    used: List[Tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        for u, v in used:
            if not (b <= u or a >= v):
                return True
        return False

    for phrase in phrases:
        needle = phrase.lower()
        if not needle:
            continue
        start = 0
        while True:
            i = lower.find(needle, start)
            if i < 0:
                break
            j = i + len(needle)
            if not overlaps(i, j):
                hits.append((i, j, phrase))
                used.append((i, j))
            start = i + max(1, len(needle))
    hits.sort(key=lambda x: x[0])
    return hits


def _entry_headers(text: str) -> List[Tuple[int, int]]:
    """Spans for Diagnoseliste entry headers like ``[#1 ...]`` or ``1. Diagnosis``."""
    spans: List[Tuple[int, int]] = []
    for match in re.finditer(r"(?m)^(?:\[[^\]]+\]|\d+\.\s+[^\n]{0,120})", text):
        spans.append((match.start(), match.end()))
    return spans


def fit_text_for_llm(
    text: str,
    task: ExtractionTask,
    *,
    max_chars: int | None = None,
) -> Tuple[str, bool]:
    """
    Return ``(text_for_llm, was_truncated)``.

    If ``len(text) <= max_chars``, returns the original text unchanged.
    """
    src = str(text or "")
    budget = max_chars if max_chars is not None else max_full_text_chars()
    if len(src) <= budget:
        return src, False

    phrases = _collect_phrases(task)
    hits = _find_hit_spans(src, phrases)
    win = _window_chars()

    keep_spans: List[Tuple[int, int]] = []
    for start, end, _phrase in hits:
        a = max(0, start - win)
        b = min(len(src), end + win)
        keep_spans.append((a, b))
    for start, end in _entry_headers(src):
        # Keep header line plus a small lead-in of the entry body.
        keep_spans.append((start, min(len(src), end + 180)))

    if not keep_spans:
        # No keywords: keep head + tail so the model still sees something useful.
        head = budget // 2
        tail = budget - head - 80
        notice = (
            f"[HINWEIS: Text von {len(src)} auf ca. {budget} Zeichen gekürzt "
            f"(keine Keyword-Treffer; Anfang + Ende behalten).]\n\n"
        )
        body = src[:head] + "\n\n[…]\n\n" + src[-max(0, tail) :]
        return (notice + body)[: budget + len(notice)], True

    # Merge overlapping spans.
    keep_spans.sort()
    merged: List[Tuple[int, int]] = []
    for a, b in keep_spans:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    notice = (
        f"[HINWEIS: Diagnoseliste von {len(src)} auf max. {budget} Zeichen gekürzt. "
        f"Nur Abschnittsköpfe und Keyword-Fenster wurden behalten.]\n\n"
    )
    parts: List[str] = [notice]
    used = len(notice)
    for a, b in merged:
        chunk = src[a:b].strip()
        if not chunk:
            continue
        block = f"[…chars {a}-{b}…]\n{chunk}\n\n"
        if used + len(block) > budget + len(notice):
            remain = budget + len(notice) - used
            if remain > 100:
                parts.append(block[:remain])
            break
        parts.append(block)
        used += len(block)

    return "".join(parts), True
