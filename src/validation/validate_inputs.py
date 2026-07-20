"""
Generic input validation.

Deterministic pre-run checks on the report source: existence, required columns,
row count, and empty/duplicate identifiers. Task-agnostic; no clinical baseline
files are referenced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.config import (
    DEFAULT_REPORTS_CSV,
    DEFAULT_REPORTS_TXT_DIR,
    REPORT_ID_COLUMN,
)
from src.preprocessing.report_loader import (
    REPORT_ID_KEY,
    REPORT_TEXT_KEY,
    load_reports,
)

LOGGER = logging.getLogger(__name__)


def run_checks(source: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return a list of ``{check, status, value, detail}`` rows."""
    rows: List[Dict[str, Any]] = []

    if source is None:
        if DEFAULT_REPORTS_CSV.exists():
            source = DEFAULT_REPORTS_CSV
        elif DEFAULT_REPORTS_TXT_DIR.exists():
            source = DEFAULT_REPORTS_TXT_DIR

    rows.append(
        {
            "check": "report_source_exists",
            "status": "ok" if source and Path(source).exists() else "fail",
            "value": str(bool(source and Path(source).exists())),
            "detail": str(source),
        }
    )
    if not source or not Path(source).exists():
        return rows

    try:
        records = load_reports(source, id_column=REPORT_ID_COLUMN)
    except Exception as exc:  # noqa: BLE001 - surfaced as a check row
        rows.append(
            {"check": "report_load", "status": "fail", "value": "error", "detail": str(exc)}
        )
        return rows

    n = len(records)
    rows.append(
        {"check": "report_row_count", "status": "ok" if n else "warn", "value": str(n), "detail": ""}
    )

    ids = [str(r.get(REPORT_ID_KEY, "")).strip() for r in records]
    missing_ids = sum(1 for i in ids if not i)
    dup_ids = len(ids) - len(set(ids))
    empty_text = sum(1 for r in records if not str(r.get(REPORT_TEXT_KEY, "")).strip())

    rows.append(
        {
            "check": "missing_report_ids",
            "status": "ok" if missing_ids == 0 else "fail",
            "value": str(missing_ids),
            "detail": "must be 0",
        }
    )
    rows.append(
        {
            "check": "duplicate_report_ids",
            "status": "ok" if dup_ids == 0 else "warn",
            "value": str(dup_ids),
            "detail": "report_id should be unique",
        }
    )
    rows.append(
        {
            "check": "empty_report_text",
            "status": "ok" if empty_text == 0 else "warn",
            "value": str(empty_text),
            "detail": "reports with no text",
        }
    )
    return rows


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Validate report inputs.")
    parser.add_argument("--reports", default=None, help="Reports CSV or .txt directory.")
    args = parser.parse_args()
    source = Path(args.reports) if args.reports else None
    rows = run_checks(source)
    print("=== Input validation ===")
    for r in rows:
        print(f"[{r['status'].upper()}] {r['check']}: {r['value']}")
        if r.get("detail"):
            print(f"    {r['detail']}")


if __name__ == "__main__":
    main()
