"""
Patient-level Diagnoseliste loader for HER-style diagnosis exports.

Expected columns (case-sensitive as exported):
  PatientID, Diagnose_Value, Diagnose_Time
Optional context columns:
  OP_TerminNummer, FallNummer, OP_BeginnZeit, OP_EndeZeit

Produces one pipeline record per PatientID whose ``report_text`` is the full
Diagnoseliste for that patient (all diagnosis rows, chronologically ordered).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from src.preprocessing.report_identity import SOURCE_ROW_ID_COL, normalize_str
from src.preprocessing.report_loader import REPORT_ID_KEY, REPORT_TEXT_KEY
from src.utils.table_io import read_table

PathLike = Union[str, Path]

LOGGER = logging.getLogger(__name__)

PATIENT_ID_ALIASES = ("PatientID", "PatientenID", "patient_id", "patientenid")
DIAGNOSE_VALUE_ALIASES = ("Diagnose_Value", "DiagnoseValue", "diagnose_value", "Diagnose")
DIAGNOSE_TIME_ALIASES = ("Diagnose_Time", "DiagnoseTime", "diagnose_time")


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    lower_map = {str(c).strip().lower(): str(c) for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise ValueError(
        f"Missing required column for {label}. Tried {aliases}. "
        f"Found columns: {list(df.columns)}"
    )


def _format_entry(idx: int, value: str, time: str, meta: dict) -> str:
    header_bits = [f"#{idx}"]
    if time:
        header_bits.append(f"time={time}")
    for key in ("FallNummer", "OP_TerminNummer", "OP_BeginnZeit", "OP_EndeZeit"):
        v = meta.get(key, "")
        if v:
            header_bits.append(f"{key}={v}")
    return f"[{' | '.join(header_bits)}]\n{value}"


def build_patient_diagnoseliste_records(df: pd.DataFrame) -> List[dict]:
    """
    Aggregate a raw HER Diagnose table into one record per patient.

    ``report_text`` is a readable Diagnoseliste block containing every diagnosis
    entry for that patient (sorted by Diagnose_Time when available).
    """
    patient_col = _find_column(df, PATIENT_ID_ALIASES, "patient id")
    value_col = _find_column(df, DIAGNOSE_VALUE_ALIASES, "diagnosis value")
    try:
        time_col = _find_column(df, DIAGNOSE_TIME_ALIASES, "diagnosis time")
    except ValueError:
        time_col = None
        LOGGER.warning("No Diagnose_Time column found; keeping source row order.")

    work = df.copy()
    work["_patient_id"] = work[patient_col].map(normalize_str)
    work["_value"] = work[value_col].map(normalize_str)
    work = work[work["_patient_id"] != ""]
    work = work[work["_value"] != ""]

    if time_col:
        work["_time_raw"] = work[time_col].map(normalize_str)
        work["_time_sort"] = pd.to_datetime(work["_time_raw"], errors="coerce")
        work = work.sort_values(["_patient_id", "_time_sort"], kind="stable", na_position="last")
    else:
        work["_time_raw"] = ""

    records: List[dict] = []
    for i, (pid, sub) in enumerate(work.groupby("_patient_id", sort=False)):
        entries: List[str] = []
        for j, (_, row) in enumerate(sub.iterrows(), start=1):
            meta = {
                k: normalize_str(row.get(k, ""))
                for k in ("FallNummer", "OP_TerminNummer", "OP_BeginnZeit", "OP_EndeZeit")
                if k in sub.columns
            }
            entries.append(
                _format_entry(j, normalize_str(row["_value"]), normalize_str(row["_time_raw"]), meta)
            )
        diagnoseliste = "[Diagnoseliste]\n\n" + "\n\n".join(entries)
        records.append(
            {
                REPORT_ID_KEY: str(pid),
                REPORT_TEXT_KEY: diagnoseliste,
                "diagnoseliste_text": diagnoseliste,
                SOURCE_ROW_ID_COL: f"patient_{i}",
                "patient_id": str(pid),
                "n_diagnosis_entries": len(entries),
                "input_kind": "patient_diagnoseliste",
            }
        )
    return records


def load_patient_diagnoseliste(path: Path) -> List[dict]:
    """Load a HER Diagnose CSV/Excel file into patient-level pipeline records."""
    return load_patient_diagnoseliste_many([path])


def load_patient_diagnoseliste_many(paths: Sequence[PathLike]) -> List[dict]:
    """
    Load one or more HER Diagnose CSV/Excel files and aggregate by PatientID.

    Rows from all files are concatenated first, then sorted by Diagnose_Time
    (when available), so a patient who appears in multiple exports gets one
    merged Diagnoseliste.
    """
    resolved: List[Path] = []
    frames: List[pd.DataFrame] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Diagnose input missing: {path}")
        df = read_table(path)
        work = df.copy()
        work["_source_file"] = path.name
        frames.append(work)
        resolved.append(path)

    if not frames:
        raise FileNotFoundError("No HER Diagnose input paths provided.")

    combined = pd.concat(frames, ignore_index=True)
    records = build_patient_diagnoseliste_records(combined)
    source_names = ", ".join(p.name for p in resolved)
    for rec in records:
        rec["source_files"] = source_names
    LOGGER.info(
        "Loaded %d patient Diagnoseliste records from %d file(s) (%d raw rows): %s",
        len(records),
        len(resolved),
        len(combined),
        source_names,
    )
    return records


def looks_like_her_diagnose_table(path: Path) -> bool:
    """Heuristic: file has PatientID + Diagnose_Value columns."""
    try:
        df = read_table(path)
    except Exception:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    has_patient = any(a.lower() in cols for a in PATIENT_ID_ALIASES)
    has_value = any(a.lower() in cols for a in DIAGNOSE_VALUE_ALIASES)
    return has_patient and has_value
