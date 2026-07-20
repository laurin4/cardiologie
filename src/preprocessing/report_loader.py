"""
Generic report loader.

A "report" is the atomic unit the pipeline consumes: ``{report_id, report_text,
<metadata...>}``. Two input shapes are supported out of the box:

1. A CSV file with a configurable id column and text column (extra columns are
   preserved as metadata). If no single text column is given, section columns can
   be stitched into one ``report_text`` block.
2. A directory of ``.txt`` files (one report per file; ``report_id`` = file stem).

No task- or hospital-specific schema is hard-coded; column names come from
``configs.config`` (overridable via environment) or explicit arguments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from configs.config import (
    DEFAULT_REPORTS_CSV,
    DEFAULT_REPORTS_TXT_DIR,
    REPORT_ID_COLUMN,
    REPORT_TEXT_COLUMN,
)
from src.preprocessing.report_identity import (
    SOURCE_ROW_ID_COL,
    assign_source_row_ids,
    normalize_str,
)
from src.utils.tabular_io import read_tabular

LOGGER = logging.getLogger(__name__)

REPORT_ID_KEY = "report_id"
REPORT_TEXT_KEY = "report_text"


def _stitch_sections(row: Dict[str, str], section_columns: Sequence[str]) -> str:
    parts: List[str] = []
    for col in section_columns:
        text = normalize_str(row.get(col, ""))
        if text:
            parts.append(f"[{col}]\n{text}")
    return "\n\n".join(parts)


def load_reports_from_csv(
    path: Path,
    *,
    id_column: str = REPORT_ID_COLUMN,
    text_column: str = REPORT_TEXT_COLUMN,
    section_columns: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Load reports from a CSV into a list of record dicts (stable ``source_row_id``)."""
    df = read_tabular(path)
    df = assign_source_row_ids(df)

    if id_column not in df.columns:
        raise ValueError(
            f"Report CSV '{path}' must contain id column '{id_column}'. "
            f"Found columns: {list(df.columns)}"
        )

    has_text_col = text_column in df.columns
    if not has_text_col and not section_columns:
        raise ValueError(
            f"Report CSV '{path}' has no text column '{text_column}' and no "
            f"section_columns were provided to stitch."
        )

    records: List[dict] = []
    for _, row in df.iterrows():
        row_dict = {c: row.get(c, "") for c in df.columns}
        rid = normalize_str(row_dict.get(id_column, ""))
        if not rid:
            continue
        if has_text_col:
            text = normalize_str(row_dict.get(text_column, ""))
        else:
            text = _stitch_sections(row_dict, section_columns or ())
        record = {
            REPORT_ID_KEY: rid,
            REPORT_TEXT_KEY: text,
            SOURCE_ROW_ID_COL: str(row_dict.get(SOURCE_ROW_ID_COL, "")),
        }
        # Preserve remaining columns as metadata (do not overwrite core keys).
        for c in df.columns:
            if c in (id_column, text_column, SOURCE_ROW_ID_COL):
                continue
            record.setdefault(c, normalize_str(row_dict.get(c, "")))
        records.append(record)
    return records


def load_reports_from_txt_dir(directory: Path) -> List[dict]:
    """Load reports from a directory of ``.txt`` files (``report_id`` = file stem)."""
    txt_files = sorted(Path(directory).glob("*.txt"))
    records: List[dict] = []
    for i, report_path in enumerate(txt_files):
        records.append(
            {
                REPORT_ID_KEY: report_path.stem,
                REPORT_TEXT_KEY: report_path.read_text(encoding="utf-8", errors="replace"),
                SOURCE_ROW_ID_COL: f"row_{i}",
            }
        )
    return records


def load_reports(source: Optional[Path] = None, **csv_kwargs) -> List[dict]:
    """
    Load reports from *source*.

    - Explicit ``.csv`` file  -> :func:`load_reports_from_csv`
    - Explicit directory      -> :func:`load_reports_from_txt_dir`
    - ``None``                -> default CSV if present, else default txt directory
    """
    if source is not None:
        source = Path(source)
        if source.is_dir():
            return load_reports_from_txt_dir(source)
        if source.suffix.lower() == ".csv":
            return load_reports_from_csv(source, **csv_kwargs)
        raise ValueError(f"Unsupported report source: {source}")

    if DEFAULT_REPORTS_CSV.exists():
        return load_reports_from_csv(DEFAULT_REPORTS_CSV, **csv_kwargs)
    if DEFAULT_REPORTS_TXT_DIR.exists():
        return load_reports_from_txt_dir(DEFAULT_REPORTS_TXT_DIR)
    raise FileNotFoundError(
        f"No report input found. Provide a CSV at {DEFAULT_REPORTS_CSV} or a "
        f"directory of .txt files at {DEFAULT_REPORTS_TXT_DIR}."
    )
