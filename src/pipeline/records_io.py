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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts (empty list if missing)."""
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def failed_keys(rows: Sequence[Dict[str, Any]], status_field: str = "status") -> Set[Tuple[str, ...]]:
    """Return resume keys for rows whose status is ``failed``."""
    return {
        resume_key(row)
        for row in rows
        if str(row.get(status_field, "")).strip().lower() == "failed"
    }


def merge_rows_by_key(
    existing: Sequence[Dict[str, Any]],
    updates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Replace existing rows with matching keys from *updates*; append new keys.

    Preserves the original order of *existing*, then appends any update keys
    that were not already present.
    """
    update_map = {resume_key(row): row for row in updates}
    used: Set[Tuple[str, ...]] = set()
    merged: List[Dict[str, Any]] = []
    for row in existing:
        key = resume_key(row)
        if key in update_map:
            merged.append(update_map[key])
            used.add(key)
        else:
            merged.append(row)
    for key, row in update_map.items():
        if key not in used:
            merged.append(row)
    return merged
