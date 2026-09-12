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


def _attach_austritt(records: List[dict], austritt_paths: Sequence[Path]) -> List[dict]:
    if not austritt_paths:
        return records
    from src.preprocessing.austritt_loader import (
        AUSTRITT_TEXT_KEY,
        load_austritt_by_fall,
        pick_austritt_for_falls,
    )

    by_fall = load_austritt_by_fall(austritt_paths)
    n_hit = 0
    for rec in records:
        falls = list(rec.get("fall_nummers") or [])
        vfall = normalize_str(rec.get("verlegung_fallnr", ""))
        if vfall and vfall not in falls:
            falls = [vfall, *falls]
        info = pick_austritt_for_falls(by_fall, falls)
        if not info:
            rec.setdefault(AUSTRITT_TEXT_KEY, "")
            continue
        n_hit += 1
        rec.update(info)
    LOGGER.info(
        "Attached Austrittsbericht via FallNummer to %d / %d records "
        "(%d Austritt FallNummer(n) available)",
        n_hit,
        len(records),
        len(by_fall),
    )
    return records


def _attach_verlegung(records: List[dict], verlegung_paths: Sequence[Path]) -> List[dict]:
    """Merge latest Verlegungsbericht onto Diagnoseliste records via FallNummer."""
    if not verlegung_paths:
        return records
    from src.preprocessing.verlegung_loader import (
        VERLEGUNG_TEXT_KEY,
        load_verlegung_by_fall,
        pick_verlegung_for_falls,
    )

    by_fall = load_verlegung_by_fall(verlegung_paths)
    n_hit = 0
    n_with_falls = 0
    for rec in records:
        falls = rec.get("fall_nummers") or []
        if falls:
            n_with_falls += 1
        info = pick_verlegung_for_falls(by_fall, falls)
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
        "Attached Verlegungsbericht via FallNummer to %d / %d patients "
        "(%d patients with FallNummer; %d Verlegung FallNummer(n) available)",
        n_hit,
        len(records),
        n_with_falls,
        len(by_fall),
    )
    if records and n_hit == 0 and by_fall:
        LOGGER.warning(
            "Verlegungsbericht FallNummer join matched 0 patients. "
            "Cohorts may not overlap (e.g. Verlegung only 2025/2026). Run: "
            "python3 scripts/check_verlegung_join.py"
        )
    return records


def filter_reports_with_verlegung(records: List[dict]) -> List[dict]:
    """Keep only patient records that already have non-empty ``verlegung_text``."""
    from src.preprocessing.verlegung_loader import VERLEGUNG_TEXT_KEY

    kept = [r for r in records if str(r.get(VERLEGUNG_TEXT_KEY) or "").strip()]
    LOGGER.info(
        "require-verlegung: kept %d / %d patients with FallNummer Verlegung match",
        len(kept),
        len(records),
    )
    return kept


def _attach_diagnose_by_fall(records: List[dict]) -> List[dict]:
    """Optional: attach Diagnoseliste text onto Verlegung-primary rows via FallNummer."""
    from src.preprocessing.diagnose_loader import load_patient_diagnoseliste_many
    from src.preprocessing.verlegung_loader import DIAGNOSELISTE_TEXT_KEY

    her_paths = discover_her_diagnose_paths()
    if not her_paths:
        return records
    patients = load_patient_diagnoseliste_many(her_paths)
    fall_to_patient: Dict[str, dict] = {}
    for p in patients:
        for f in p.get("fall_nummers") or []:
            key = normalize_str(f)
            if key and key not in fall_to_patient:
                fall_to_patient[key] = p
    n_hit = 0
    for rec in records:
        fall = normalize_str(rec.get("verlegung_fallnr") or rec.get(REPORT_ID_KEY, ""))
        p = fall_to_patient.get(fall)
        if not p:
            rec.setdefault(DIAGNOSELISTE_TEXT_KEY, "")
            continue
        n_hit += 1
        rec[DIAGNOSELISTE_TEXT_KEY] = (
            p.get(DIAGNOSELISTE_TEXT_KEY) or p.get(REPORT_TEXT_KEY) or ""
        )
        if p.get("patient_id"):
            rec["patient_id"] = p["patient_id"]
        rec["input_kind"] = "patient_verlegung+diagnoseliste"
    LOGGER.info(
        "Attached Diagnoseliste via FallNummer to %d / %d Verlegung-primary records",
        n_hit,
        len(records),
    )
    return records


def _finalize_side_sources(
    records: List[dict], *, verlegung_paths: Optional[Sequence[Path]] = None
) -> List[dict]:
    from src.preprocessing.austritt_loader import discover_her_austritt_paths
    from src.preprocessing.verlegung_loader import discover_her_verlegung_paths

    if verlegung_paths is not None:
        records = _attach_verlegung(records, list(verlegung_paths))
    records = _attach_austritt(records, discover_her_austritt_paths())
    return records


def load_reports(source: SourceArg = None, **table_kwargs) -> List[dict]:
    """
    Load reports from *source*.

    - Explicit directory           -> txt files
    - HER Diagnose CSV/Excel       -> patient Diagnoseliste; merge IPS Verlegung
      2025 + Austrittsbericht via FallNummer
    - HER IPS Verlegung only       -> one row per FallNummer; attach Diagnose + Austritt
    - ``None``                     -> all HER_Diagnose* (+ IPS Verlegung + Austritt)
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
        verlegung_paths = discover_her_verlegung_paths()  # IPS-only by default
        her_paths = discover_her_diagnose_paths()
        # Prefer IPS-Verlegung primary (larger Dendrite FallNummer overlap).
        if verlegung_paths:
            LOGGER.info(
                "Using %d default HER IPS Verlegung input(s) (primary): %s",
                len(verlegung_paths),
                ", ".join(p.name for p in verlegung_paths),
            )
            return load_reports(verlegung_paths)
        if her_paths:
            LOGGER.info(
                "Using %d default HER Diagnose input(s): %s",
                len(her_paths),
                ", ".join(p.name for p in her_paths),
            )
            records = load_patient_diagnoseliste_many(her_paths)
            return _finalize_side_sources(records, verlegung_paths=verlegung_paths)
        if DEFAULT_REPORTS_CSV.exists():
            return load_reports_from_table(DEFAULT_REPORTS_CSV, **table_kwargs)
        if DEFAULT_REPORTS_TXT_DIR.exists():
            return load_reports_from_txt_dir(DEFAULT_REPORTS_TXT_DIR)
        raise FileNotFoundError(
            f"No report input found. Place HER_IPS_Verlegungsbericht / HER_Diagnose "
            f"under {RAW_DATA_DIR}, a CSV at {DEFAULT_REPORTS_CSV}, or .txt files at "
            f"{DEFAULT_REPORTS_TXT_DIR}."
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
        attach = verlegung_paths or discover_her_verlegung_paths()
        return _finalize_side_sources(records, verlegung_paths=attach)

    if verlegung_paths and not her_paths:
        from src.preprocessing.verlegung_loader import (
            VERLEGUNG_TEXT_KEY,
            load_verlegung_by_fall,
        )

        by_fall = load_verlegung_by_fall(verlegung_paths)
        records = []
        for i, (fall, info) in enumerate(by_fall.items()):
            text = info.get(VERLEGUNG_TEXT_KEY, "")
            clean = {k: v for k, v in info.items() if not str(k).startswith("_")}
            records.append(
                {
                    REPORT_ID_KEY: fall,
                    REPORT_TEXT_KEY: text,
                    "diagnoseliste_text": "",
                    SOURCE_ROW_ID_COL: f"fall_{i}",
                    "patient_id": clean.get("verlegung_patnr") or fall,
                    "fall_nummers": [fall],
                    "input_kind": "patient_verlegung",
                    **clean,
                }
            )
        records = _attach_diagnose_by_fall(records)
        return _finalize_side_sources(records, verlegung_paths=None)

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
