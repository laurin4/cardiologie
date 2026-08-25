"""
Loader for HER Verlegungsbericht (transfer report) exports.

Expected columns (semicolon CSV / Excel):
  OP_TerminNummer, FallNummer, OP_BeginnZeit, OP_EndeZeit,
  patnr, fallnr, berdat, bertyp, ber, name,
  diag, epikrise, jetziges_leiden, prozedere

Join key to Diagnoseliste: **FallNummer** / ``fallnr`` (clinic guidance).

Produces one record per FallNummer: the **latest** Verlegungsbericht by ``berdat``,
with section-labelled text from diag / epikrise / jetziges_leiden / prozedere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from src.preprocessing.report_identity import normalize_str
from src.utils.table_io import read_table

PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)

PATIENT_ALIASES = ("patnr", "PatientID", "PatientenID", "patient_id", "patientenid")
FALL_ALIASES = ("FallNummer", "fallnr", "Fallnummer", "fall_nummer", "Fall Nr", "FallNr")
BERDAT_ALIASES = ("berdat", "BerichtDatum", "bericht_datum")
SECTION_COLS = ("diag", "epikrise", "jetziges_leiden", "prozedere")

VERLEGUNG_TEXT_KEY = "verlegung_text"
DIAGNOSELISTE_TEXT_KEY = "diagnoseliste_text"


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


def looks_like_her_verlegung_table(path: Path) -> bool:
    """Heuristic: has FallNummer (or patnr) + at least one clinical section column."""
    try:
        df = read_table(path)
    except Exception:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    has_fall = any(a.lower() in cols for a in FALL_ALIASES)
    has_patient = any(a.lower() in cols for a in PATIENT_ALIASES)
    has_section = any(s.lower() in cols for s in SECTION_COLS)
    return (has_fall or has_patient) and has_section


def _format_verlegung_text(row: pd.Series) -> str:
    parts: List[str] = []
    meta_bits = []
    for key in ("FallNummer", "fallnr", "OP_TerminNummer", "berdat", "bertyp", "patnr"):
        if key in row.index:
            v = normalize_str(row.get(key, ""))
            if v:
                meta_bits.append(f"{key}={v}")
    header = "[Verlegungsbericht]"
    if meta_bits:
        header += " " + " | ".join(meta_bits)
    parts.append(header)
    for col in SECTION_COLS:
        actual = None
        for c in row.index:
            if str(c).strip().lower() == col:
                actual = c
                break
        if actual is None:
            continue
        text = normalize_str(row.get(actual, ""))
        if text:
            parts.append(f"[{col}]\n{text}")
    return "\n\n".join(parts)


def build_latest_verlegung_by_fall(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Return ``{fall_nummer: {verlegung_text, berdat, patnr, ...}}`` using the
    latest ``berdat`` row per FallNummer.
    """
    fall_col = _find_column(df, FALL_ALIASES, "FallNummer / fallnr")
    try:
        berdat_col = _find_column(df, BERDAT_ALIASES, "berdat")
    except ValueError:
        berdat_col = None
        LOGGER.warning("No berdat column; keeping last row per FallNummer in file order.")

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
        row = sub.iloc[-1]  # latest after sort
        text = _format_verlegung_text(row)
        out[str(fall)] = {
            VERLEGUNG_TEXT_KEY: text,
            "verlegung_berdat": normalize_str(row.get(berdat_col, "")) if berdat_col else "",
            "verlegung_fallnr": str(fall),
            "verlegung_patnr": (
                normalize_str(row.get(patient_col, "")) if patient_col else ""
            ),
            "n_verlegung_rows": int(len(sub)),
            "_berdat_sort": row.get("_berdat_sort"),
        }
    return out


# Backwards-compatible alias (old name keyed wrongly by patient).
def build_latest_verlegung_by_patient(df: pd.DataFrame) -> Dict[str, dict]:
    """Deprecated name: now indexes by FallNummer."""
    return build_latest_verlegung_by_fall(df)


def load_verlegung_by_fall(paths: Sequence[PathLike]) -> Dict[str, dict]:
    """Load one or more Verlegungsbericht files; keep latest berdat per FallNummer."""
    frames: List[pd.DataFrame] = []
    resolved: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Verlegungsbericht input missing: {path}")
        df = read_table(path)
        work = df.copy()
        work["_source_file"] = path.name
        frames.append(work)
        resolved.append(path)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    by_fall = build_latest_verlegung_by_fall(combined)
    LOGGER.info(
        "Loaded latest Verlegungsbericht for %d FallNummer(n) from %d file(s) (%d raw rows): %s",
        len(by_fall),
        len(resolved),
        len(combined),
        ", ".join(p.name for p in resolved),
    )
    return by_fall


def load_verlegung_by_patient(paths: Sequence[PathLike]) -> Dict[str, dict]:
    """Deprecated name: now indexes by FallNummer."""
    return load_verlegung_by_fall(paths)


def pick_verlegung_for_falls(
    by_fall: Dict[str, dict], fall_nummers: Sequence[str]
) -> Optional[dict]:
    """
    Among FallNummern on a Diagnoseliste patient, pick the matching Verlegung
    with the latest ``berdat`` (clinic: join on Fallnummer).
    """
    matches: List[dict] = []
    for fall in fall_nummers:
        key = normalize_str(fall)
        if not key:
            continue
        info = by_fall.get(key)
        if info:
            matches.append(info)
    if not matches:
        return None
    if len(matches) == 1:
        chosen = dict(matches[0])
    else:
        def _sort_key(item: dict):
            ts = item.get("_berdat_sort")
            if ts is None or (isinstance(ts, float) and pd.isna(ts)):
                return pd.Timestamp.min
            return ts

        chosen = dict(max(matches, key=_sort_key))
    chosen.pop("_berdat_sort", None)
    chosen["n_fall_matches"] = len(matches)
    return chosen


def discover_her_verlegung_paths(raw_dir: Optional[Path] = None) -> List[Path]:
    from configs.config import RAW_DATA_DIR
    from src.utils.table_io import is_excel_path

    root = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    if not root.exists():
        return []
    suffixes = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}
    candidates = sorted(root.glob("HER_Verlegungsbericht*"))
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


def select_text_for_source(report: dict, text_source: str) -> str:
    """
    Pick patient text for a variable's ``text_source``.

    ``report`` may contain ``diagnoseliste_text``, ``verlegung_text``, and/or
    legacy ``report_text``.
    """
    diag = normalize_str(report.get(DIAGNOSELISTE_TEXT_KEY, ""))
    verl = normalize_str(report.get(VERLEGUNG_TEXT_KEY, ""))
    legacy = normalize_str(report.get("report_text", ""))
    if not diag and legacy and "[Verlegungsbericht]" not in legacy:
        diag = legacy
    if not verl and legacy and "[Verlegungsbericht]" in legacy and not diag:
        verl = legacy

    src = (text_source or "report").strip().lower()
    if src == "diagnoseliste":
        return diag or legacy
    if src == "verlegung":
        return verl
    if src == "both":
        parts = [p for p in (diag, verl) if p]
        if parts:
            return "\n\n".join(parts)
        return legacy
    # legacy / default
    return legacy or diag or verl
