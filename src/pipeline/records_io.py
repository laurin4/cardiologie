"""
Result IO helpers for the extraction pipeline.

Writes the canonical flattened CSV and a structured JSONL of final objects, and
provides resume/checkpoint helpers so long runs can continue after interruption.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

RESUME_ID_KEYS = ("source_row_id", "report_id")


def write_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: Sequence[str]) -> None:
    """Write *rows* to *path* as CSV using *fieldnames* (extra keys ignored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write *records* to *path*, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def resume_key(mapping: Dict[str, Any]) -> Tuple[str, ...]:
    """Stable identity key for a report/result row (source_row_id, else report_id)."""
    for key in RESUME_ID_KEYS:
        val = str(mapping.get(key, "") or "").strip()
        if val and val.lower() not in ("nan", "none"):
            return (key, val)
    return ("idx", str(mapping))


def load_checkpoint_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def done_keys(rows: Sequence[Dict[str, Any]]) -> Set[Tuple[str, ...]]:
    return {resume_key(row) for row in rows}


def filter_unprocessed(reports: Sequence[dict], done: Set[Tuple[str, ...]]) -> List[dict]:
    return [r for r in reports if resume_key(r) not in done]
