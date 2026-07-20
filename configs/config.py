"""
Central configuration for the clinical extraction framework.

Single source of truth for filesystem paths and a few run-level settings. Values
are intentionally generic (no task-specific constants live here). Task-specific
configuration lives under ``configs/tasks/`` (see :mod:`configs.tasks`).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Project layout --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "extractions"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
LOGS_DIR = OUTPUTS_DIR / "logs"
LLM_DEBUG_DIR = LOGS_DIR / "llm_debug"

PROMPTS_DIR = PROJECT_ROOT / "prompts"
TASKS_DIR = PROJECT_ROOT / "configs" / "tasks"

SQLITE_PREDICTIONS_DB_PATH = LOGS_DIR / "extractions.sqlite"

# --- Report input (generic) ------------------------------------------------
# A "report" is just {id, text, metadata}. Inputs can be a CSV (configurable
# id/text columns) or a directory of ``.txt`` files. Override via environment.
DEFAULT_REPORTS_CSV = RAW_DATA_DIR / "reports.csv"
DEFAULT_REPORTS_TXT_DIR = RAW_DATA_DIR / "reports"

REPORT_ID_COLUMN = os.getenv("REPORT_ID_COLUMN", "report_id")
REPORT_TEXT_COLUMN = os.getenv("REPORT_TEXT_COLUMN", "report_text")

# Default extraction task (name resolved by configs.tasks.load_task).
DEFAULT_TASK = os.getenv("EXTRACTION_TASK", "demo_extraction")


# --- Run size cap ----------------------------------------------------------
DEFAULT_MAX_REPORTS: int | None = None


def parse_max_reports_env(raw: str | None = None) -> int | None:
    """
    Parse ``MAX_REPORTS`` from *raw* or the environment.

    - Unset / blank -> ``DEFAULT_MAX_REPORTS`` (``None`` = process everything).
    - ``all`` (case-insensitive) -> ``None`` (no limit).
    - Otherwise a positive integer cap.

    Raises ``ValueError`` for invalid strings or non-positive integers.
    """
    if raw is None:
        raw = os.environ.get("MAX_REPORTS", "")
    raw = raw.strip()
    if not raw:
        return DEFAULT_MAX_REPORTS
    if raw.lower() == "all":
        return None
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MAX_REPORTS must be 'all', a positive integer, or unset (process all)."
        ) from exc
    if n <= 0:
        raise ValueError(
            "MAX_REPORTS must be 'all', a positive integer, or unset (process all)."
        )
    return n


MAX_REPORTS = parse_max_reports_env()
