"""
Generic report loader.

A "report" is the atomic unit the pipeline consumes: ``{report_id, report_text,
<metadata...>}``. Supported input shapes:

1. A CSV/Excel with a configurable id column and text column.
2. One or more HER-style Diagnose tables (PatientID + Diagnose_Value) aggregated
   to one Diagnoseliste per patient (files are merged by PatientID).
3. A directory of ``.txt`` files (one report per file; ``report_id`` = file stem).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

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

SourceArg = Optional[Union[Path, str, Sequence[Union[Path, str]]]]


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


def discover_her_diagnose_paths(raw_dir: Optional[Path] = None) -> List[Path]:
    """
    Find all HER_Diagnose* tabular files under data/raw/.

    Prefers CSV when both CSV and Excel exist for the same stem; otherwise keeps
    every distinct path (e.g. HER_Diagnose_202601_202606.csv + HER_Diagnose_vor2026.xlsx).
    """
    root = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    if not root.exists():
        return []
    candidates = sorted(root.glob("HER_Diagnose*"))
    # Do not pick up Verlegungsbericht files that happen to match a broad glob.
    candidates = [p for p in candidates if "verlegung" not in p.name.lower()]
    tabular = [
        p
        for p in candidates
        if p.is_file() and (p.suffix.lower() in _TABULAR_SUFFIXES or is_excel_path(p))
    ]
    # Prefer .csv over excel for the same stem.
    by_stem: Dict[str, Path] = {}
    for path in tabular:
        stem = path.stem
        prev = by_stem.get(stem)
        if prev is None:
            by_stem[stem] = path
            continue
        if prev.suffix.lower() != ".csv" and path.suffix.lower() == ".csv":
            by_stem[stem] = path
    return sorted(by_stem.values(), key=lambda p: p.name.lower())


def _normalize_sources(source: SourceArg) -> List[Path]:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        return [Path(source)]
    return [Path(p) for p in source]


def _attach_verlegung(records: List[dict], verlegung_paths: Sequence[Path]) -> List[dict]:
    """Merge latest Verlegungsbericht text onto Diagnoseliste patient records."""
    if not verlegung_paths:
        return records
    from src.preprocessing.verlegung_loader import (
        VERLEGUNG_TEXT_KEY,
        load_verlegung_by_patient,
    )

    by_patient = load_verlegung_by_patient(verlegung_paths)
    n_hit = 0
    for rec in records:
        pid = normalize_str(rec.get("patient_id") or rec.get(REPORT_ID_KEY) or "")
        info = by_patient.get(pid)
        if not info:
            rec.setdefault(VERLEGUNG_TEXT_KEY, "")
            continue
        n_hit += 1
        rec.update(info)
        # Keep report_text as Diagnoseliste; combined text is built per-variable.
        src = str(rec.get("source_files") or "")
        extra = ", ".join(Path(p).name for p in verlegung_paths)
        rec["source_files"] = f"{src}; {extra}" if src else extra
        rec["input_kind"] = "patient_diagnoseliste+verlegung"
    LOGGER.info(
        "Attached Verlegungsbericht to %d / %d patients (%d verlegung patients available)",
        n_hit,
        len(records),
        len(by_patient),
    )
    if records and n_hit == 0 and by_patient:
        LOGGER.warning(
            "Verlegungsbericht join matched 0 patients. Check PatientID vs patnr "
            "(formats / leading zeros / Excel .0). Run: "
            "python3 scripts/check_verlegung_join.py"
        )
    return records


def load_reports(source: SourceArg = None, **table_kwargs) -> List[dict]:
    """
    Load reports from *source*.

    - Explicit directory           -> txt files
    - HER Diagnose CSV/Excel       -> patient Diagnoseliste; optionally merge
      HER_Verlegungsbericht* (latest by berdat) onto the same patients
    - ``None``                     -> all HER_Diagnose* (+ Verlegung) under data/raw/
    """
    from src.preprocessing.diagnose_loader import (
        load_patient_diagnoseliste_many,
        looks_like_her_diagnose_table,
    )
    from src.preprocessing.verlegung_loader import (
        discover_her_verlegung_paths,
        looks_like_her_verlegung_table,
    )

    paths = _normalize_sources(source)

    if not paths:
        her_paths = discover_her_diagnose_paths()
        verlegung_paths = discover_her_verlegung_paths()
        if her_paths:
            LOGGER.info(
                "Using %d default HER Diagnose input(s): %s",
                len(her_paths),
                ", ".join(p.name for p in her_paths),
            )
            records = load_patient_diagnoseliste_many(her_paths)
            return _attach_verlegung(records, verlegung_paths)
        if DEFAULT_REPORTS_CSV.exists():
            return load_reports_from_table(DEFAULT_REPORTS_CSV, **table_kwargs)
        if DEFAULT_REPORTS_TXT_DIR.exists():
            return load_reports_from_txt_dir(DEFAULT_REPORTS_TXT_DIR)
        raise FileNotFoundError(
            f"No report input found. Place HER_Diagnose CSV/Excel under {RAW_DATA_DIR}, "
            f"a CSV at {DEFAULT_REPORTS_CSV}, or .txt files at {DEFAULT_REPORTS_TXT_DIR}."
        )

    if len(paths) == 1 and paths[0].is_dir():
        return load_reports_from_txt_dir(paths[0])

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Report input missing: {path}")

    her_paths = [
        p
        for p in paths
        if (p.suffix.lower() in _TABULAR_SUFFIXES or is_excel_path(p))
        and looks_like_her_diagnose_table(p)
    ]
    verlegung_paths = [
        p
        for p in paths
        if (p.suffix.lower() in _TABULAR_SUFFIXES or is_excel_path(p))
        and looks_like_her_verlegung_table(p)
        and p not in her_paths
    ]
    other_paths = [p for p in paths if p not in her_paths and p not in verlegung_paths]

    if other_paths and (her_paths or verlegung_paths):
        raise ValueError(
            "Cannot mix HER clinical tables with other report sources in one call. "
            f"other={ [p.name for p in other_paths] }"
        )

    if her_paths:
        LOGGER.info(
            "Loading %d HER Diagnose file(s): %s",
            len(her_paths),
            ", ".join(p.name for p in her_paths),
        )
        records = load_patient_diagnoseliste_many(her_paths)
        # If caller did not pass Verlegung files, still auto-attach from data/raw/.
        attach = verlegung_paths or discover_her_verlegung_paths()
        return _attach_verlegung(records, attach)

    if verlegung_paths and not her_paths:
        # Verlegung-only: synthesize patient records from Verlegung text.
        from src.preprocessing.verlegung_loader import (
            VERLEGUNG_TEXT_KEY,
            load_verlegung_by_patient,
        )

        by_patient = load_verlegung_by_patient(verlegung_paths)
        records = []
        for i, (pid, info) in enumerate(by_patient.items()):
            text = info.get(VERLEGUNG_TEXT_KEY, "")
            records.append(
                {
                    REPORT_ID_KEY: pid,
                    REPORT_TEXT_KEY: text,
                    "diagnoseliste_text": "",
                    SOURCE_ROW_ID_COL: f"patient_{i}",
                    "patient_id": pid,
                    "input_kind": "patient_verlegung",
                    **info,
                }
            )
        return records

    if len(paths) == 1:
        path = paths[0]
        if path.suffix.lower() in _TABULAR_SUFFIXES or is_excel_path(path):
            return load_reports_from_table(path, **table_kwargs)
        raise ValueError(f"Unsupported report source: {path}")

    records = []
    for path in paths:
        if path.suffix.lower() in _TABULAR_SUFFIXES or is_excel_path(path):
            records.extend(load_reports_from_table(path, **table_kwargs))
        else:
            raise ValueError(f"Unsupported report source: {path}")
    return records
