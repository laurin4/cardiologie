"""
Generic section splitting for clinical reports.

Sections are defined per task as ``(marker, label)`` pairs (see
:class:`configs.tasks.base.ExtractionTask`). Text before the first marker is
labelled ``unknown``. This module is the single source of truth for section
ranges, reused by the rule-based evidence extractor.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

SectionRange = Tuple[int, int, str]  # (start, end, label)


def build_section_ranges(
    text: str, markers: Tuple[Tuple[str, str], ...]
) -> List[SectionRange]:
    """
    Return non-overlapping ``(start, end, label)`` ranges covering *text*.

    If no markers are supplied or none are found, the whole text is one
    ``unknown`` range.
    """
    if not text:
        return [(0, 0, "unknown")]
    if not markers:
        return [(0, len(text), "unknown")]

    hits: List[Tuple[int, str]] = []
    for marker, label in markers:
        if not marker:
            continue
        pos = 0
        mlen = len(marker)
        while True:
            i = text.find(marker, pos)
            if i < 0:
                break
            hits.append((i, label))
            pos = i + mlen

    if not hits:
        return [(0, len(text), "unknown")]

    hits.sort(key=lambda x: x[0])
    spans: List[SectionRange] = []
    if hits[0][0] > 0:
        spans.append((0, hits[0][0], "unknown"))
    for j, (pos, label) in enumerate(hits):
        end = hits[j + 1][0] if j + 1 < len(hits) else len(text)
        spans.append((pos, end, label))
    return spans


def section_for_index(section_ranges: List[SectionRange], idx: int) -> str:
    """Return the section label containing character index *idx*."""
    for start, end, label in section_ranges:
        if start <= idx < end:
            return label
    return "unknown"


def split_sections(
    text: str, markers: Tuple[Tuple[str, str], ...]
) -> Dict[str, str]:
    """Return a ``{label: text}`` mapping (labels concatenated if repeated)."""
    out: Dict[str, str] = {}
    for start, end, label in build_section_ranges(text, markers):
        chunk = text[start:end].strip()
        if not chunk:
            continue
        out[label] = (out.get(label, "") + "\n" + chunk).strip() if label in out else chunk
    return out
