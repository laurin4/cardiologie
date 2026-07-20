"""
Optional SQLite append-log for extraction result rows (CSV remains canonical).

Enable with: ENABLE_SQLITE_LOGGING=true
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping


def init_extraction_db(path: Path | str) -> None:
    """Create parent dirs and ensure the extraction results table exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_extraction_rows_record ON extraction_rows (record_id)"
        )
        conn.commit()
    finally:
        conn.close()


def log_extraction_row(
    path: Path | str, row_dict: Mapping[str, Any], id_field: str = "record_id"
) -> None:
    """Insert one result row as a JSON payload (UTF-8, ``default=str`` for odd types)."""
    p = Path(path)
    rid = str(row_dict.get(id_field, "") or "")
    payload = json.dumps(dict(row_dict), ensure_ascii=False, default=str)
    conn = sqlite3.connect(str(p))
    try:
        conn.execute(
            "INSERT INTO extraction_rows (record_id, payload) VALUES (?, ?)",
            (rid, payload),
        )
        conn.commit()
    finally:
        conn.close()
