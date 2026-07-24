"""
Generic report loader.

A "report" is the atomic unit the pipeline consumes: ``{report_id, report_text,
<metadata...>}``. Supported input shapes:

1. A CSV/Excel with a configurable id column and text column.
2. A HER-style Diagnose table (PatientID + Diagnose_Value) aggregated to one
   Diagnoseliste per patient.
3. A directory of ``.txt`` files (one report per file; ``report_id`` = file stem).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from configs.config import (
    DEFAULT_REPORTS_CSV,
    DEFAULT_REPORTS_TXT_DIR,
    RAW_DATA_DIR,
    REPORT_ID_COLUMN,
    REPORT_TEXT_COLUMN,
)
from src.preprocessing.report_identity import (
    SOURCE_ROW_ID_COL,
    assign_source_row_ids,
    normalize_str,
)
from src.utils.table_io import is_excel_path, read_table

LOGGER = logging.getLogger(__name__)

REPORT_ID_KEY = "report_id"
REPORT_TEXT_KEY = "report_text"

_TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}


def _stitch_sections(row: Dict[str, str], section_columns: Sequence[str]) -> str:
    parts: List[str] = []
    for col in section_columns:
        text = normalize_str(row.get(col, ""))
        if text:
            parts.append(f"[{col}]\n{text}")
    return "\n\n".join(parts)


def load_reports_from_table(
    path: Path,
    *,
    id_column: str = REPORT_ID_COLUMN,
    text_column: str = REPORT_TEXT_COLUMN,
    section_columns: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Load one-row-per-report records from a CSV/Excel file."""
    from src.preprocessing.diagnose_loader import (
        build_patient_diagnoseliste_records,
        looks_like_her_diagnose_table,
    )

    # HER Diagnose exports: aggregate to one Diagnoseliste per patient.
    if looks_like_her_diagnose_table(path):
        LOGGER.info("Detected HER-style Diagnose table; aggregating by PatientID.")
        return build_patient_diagnoseliste_records(read_table(path))

    df = read_table(path)
    df = assign_source_row_ids(df)

    if id_column not in df.columns:
        raise ValueError(
            f"Report table '{path}' must contain id column '{id_column}'. "
            f"Found columns: {list(df.columns)}"
        )

    has_text_col = text_column in df.columns
    if not has_text_col and not section_columns:
        raise ValueError(
            f"Report table '{path}' has no text column '{text_column}' and no "
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
        for c in df.columns:
            if c in (id_column, text_column, SOURCE_ROW_ID_COL):
                continue
            record.setdefault(c, normalize_str(row_dict.get(c, "")))
        records.append(record)
    return records


# Backwards-compatible alias used by existing tests.
def load_reports_from_csv(
    path: Path,
    *,
    id_column: str = REPORT_ID_COLUMN,
    text_column: str = REPORT_TEXT_COLUMN,
    section_columns: Optional[Sequence[str]] = None,
) -> List[dict]:
    return load_reports_from_table(
        path,
        id_column=id_column,
        text_column=text_column,
        section_columns=section_columns,
    )


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


def _default_her_diagnose_path() -> Optional[Path]:
    """Prefer a HER_Diagnose* file under data/raw/ when present."""
    if not RAW_DATA_DIR.exists():
        return None
    candidates = sorted(RAW_DATA_DIR.glob("HER_Diagnose*"))
    for path in candidates:
        if path.suffix.lower() in _TABULAR_SUFFIXES or is_excel_path(path):
            return path
    return None


def load_reports(source: Optional[Path] = None, **table_kwargs) -> List[dict]:
    """
    Load reports from *source*.

    - Explicit directory           -> txt files
    - Explicit CSV/Excel           -> table loader (auto-detects HER Diagnose)
    - ``None``                     -> HER_Diagnose* in data/raw, else default CSV/txt
    """
    if source is not None:
        source = Path(source)
        if source.is_dir():
            return load_reports_from_txt_dir(source)
        if source.suffix.lower() in _TABULAR_SUFFIXES or is_excel_path(source):
            return load_reports_from_table(source, **table_kwargs)
        raise ValueError(f"Unsupported report source: {source}")

    her_path = _default_her_diagnose_path()
    if her_path is not None:
        LOGGER.info("Using default HER Diagnose input: %s", her_path)
        return load_reports_from_table(her_path, **table_kwargs)
    if DEFAULT_REPORTS_CSV.exists():
        return load_reports_from_table(DEFAULT_REPORTS_CSV, **table_kwargs)
    if DEFAULT_REPORTS_TXT_DIR.exists():
        return load_reports_from_txt_dir(DEFAULT_REPORTS_TXT_DIR)
    raise FileNotFoundError(
        f"No report input found. Place a HER_Diagnose CSV/Excel under {RAW_DATA_DIR}, "
        f"a CSV at {DEFAULT_REPORTS_CSV}, or .txt files at {DEFAULT_REPORTS_TXT_DIR}."
    )
