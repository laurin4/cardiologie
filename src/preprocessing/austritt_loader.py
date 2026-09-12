"""
Loader for HER Austrittsbericht (discharge / admission-status) exports.

Expected columns (semicolon CSV / Excel), truncated names allowed:
  OP_TerminN, FallNummer, OP_BeginnZ, OP_EndeZei, OP_Zeit,
  patnr, fallnr, berdat, bername, anamn|anamnese, stat_ein

Join key: **FallNummer** / ``fallnr``.

Produces one record per FallNummer: latest by ``berdat``, with section-labelled
text from ``stat_ein`` (primary for cirrhosis) and ``anamn``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.preprocessing.report_identity import normalize_str
from src.preprocessing.verlegung_loader import (
    BERDAT_ALIASES,
    FALL_ALIASES,
    PATIENT_ALIASES,
    _find_column,
    _resolve_section_column,
)
from src.utils.table_io import is_excel_path, read_table

PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)

SECTION_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("stat_ein", ("stat_ein", "status_eintritt", "status_bei_eintritt")),
    ("anamn", ("anamn", "anamnese", "anamnes")),
)
_ALL_SECTION_ALIASES = tuple(a for _, aliases in SECTION_ALIASES for a in aliases)

AUSTRITT_TEXT_KEY = "austritt_text"


def looks_like_her_austritt_table(path: Path) -> bool:
    try:
        df = read_table(path)
    except Exception:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    has_fall = any(a.lower() in cols for a in FALL_ALIASES)
    has_section = any(s.lower() in cols for s in _ALL_SECTION_ALIASES)
    name_hit = "austritt" in path.name.lower()
    return has_fall and (has_section or name_hit)


def _format_austritt_text(row: pd.Series) -> str:
    parts: List[str] = []
    meta_bits = []
    for key in ("FallNummer", "fallnr", "OP_TerminN", "OP_TerminNummer", "berdat", "patnr"):
        if key in row.index:
            v = normalize_str(row.get(key, ""))
            if v:
                meta_bits.append(f"{key}={v}")
    header = "[Austrittsbericht]"
    if meta_bits:
        header += " " + " | ".join(meta_bits)
    parts.append(header)
    for canonical, aliases in SECTION_ALIASES:
        actual = _resolve_section_column(row, aliases)
        if actual is None:
            continue
        text = normalize_str(row.get(actual, ""))
        if text:
            parts.append(f"[{canonical}]\n{text}")
    return "\n\n".join(parts)


def build_latest_austritt_by_fall(df: pd.DataFrame) -> Dict[str, dict]:
    fall_col = _find_column(df, FALL_ALIASES, "FallNummer / fallnr")
    try:
        berdat_col = _find_column(df, BERDAT_ALIASES, "berdat")
    except ValueError:
        berdat_col = None
        LOGGER.warning("No berdat on Austritt; keeping last row per FallNummer.")

    patient_col = None
    try:
        patient_col = _find_column(df, PATIENT_ALIASES, "patient id (patnr)")
    except ValueError:
        pass

    work = df.copy()
    work["_fall_nummer"] = work[fall_col].map(normalize_str)
    work = work[work["_fall_nummer"] != ""]

    if berdat_col:
        work["_berdat_sort"] = pd.to_datetime(
            work[berdat_col].map(normalize_str), errors="coerce"
        )
        work = work.sort_values(
            ["_fall_nummer", "_berdat_sort"], kind="stable", na_position="first"
        )
    else:
        work["_berdat_sort"] = pd.NaT

    out: Dict[str, dict] = {}
    for fall, sub in work.groupby("_fall_nummer", sort=False):
        row = sub.iloc[-1]
        out[str(fall)] = {
            AUSTRITT_TEXT_KEY: _format_austritt_text(row),
            "austritt_berdat": normalize_str(row.get(berdat_col, "")) if berdat_col else "",
            "austritt_fallnr": str(fall),
            "austritt_patnr": (
                normalize_str(row.get(patient_col, "")) if patient_col else ""
            ),
            "n_austritt_rows": int(len(sub)),
        }
    return out


def load_austritt_by_fall(paths: Sequence[PathLike]) -> Dict[str, dict]:
    frames: List[pd.DataFrame] = []
    resolved: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Austrittsbericht input missing: {path}")
        df = read_table(path)
        work = df.copy()
        work["_source_file"] = path.name
        frames.append(work)
        resolved.append(path)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    by_fall = build_latest_austritt_by_fall(combined)
    LOGGER.info(
        "Loaded latest Austrittsbericht for %d FallNummer(n) from %d file(s) (%d raw rows): %s",
        len(by_fall),
        len(resolved),
        len(combined),
        ", ".join(p.name for p in resolved),
    )
    return by_fall


def pick_austritt_for_falls(
    by_fall: Dict[str, dict], fall_nummers: Sequence[str]
) -> Optional[dict]:
    for fall in fall_nummers:
        key = normalize_str(fall)
        if key and key in by_fall:
            return dict(by_fall[key])
    return None


def discover_her_austritt_paths(raw_dir: Optional[Path] = None) -> List[Path]:
    from configs.config import RAW_DATA_DIR

    root = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    if not root.exists():
        return []
    suffixes = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}
    candidates = sorted(
        {
            *root.glob("HER_Austrittsbericht*"),
            *root.glob("HER_*Austrittsbericht*"),
        }
    )
    tabular = [
        p
        for p in candidates
        if p.is_file() and (p.suffix.lower() in suffixes or is_excel_path(p))
    ]
    by_stem: Dict[str, Path] = {}
    for path in tabular:
        stem = path.stem
        prev = by_stem.get(stem)
        if prev is None or (prev.suffix.lower() != ".csv" and path.suffix.lower() == ".csv"):
            by_stem[stem] = path
    return sorted(by_stem.values(), key=lambda p: p.name.lower())
