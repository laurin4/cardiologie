"""
Loader for HER Re-OP exports (follow-up operation reports after the index OP).

Expected columns (semicolon CSV / Excel):
  FallNummer, FallKey, Index_OP_TerminNummer, Index_OPNw_Eingriff,
  berdat, bername, operation, zugang

One row = one follow-up OP report. The export contains **only** cases with at
least one follow-up OP, so a FallNummer that is absent from a loaded export
means "no re-operation in this export" (see ``REOP_LOADED_KEY``).

Join key: **FallNummer**. Produces one record per FallNummer with all follow-up
OPs in chronological order (``berdat``), section-labelled by source column.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from src.preprocessing.report_identity import normalize_str
from src.preprocessing.verlegung_loader import BERDAT_ALIASES, FALL_ALIASES, _find_column
from src.utils.table_io import is_excel_path, read_table

PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)

REOP_TEXT_KEY = "reop_text"
# Set on every cohort record when a Re-OP export was loaded (match or not).
REOP_LOADED_KEY = "reop_source_loaded"

INDEX_OP_ALIASES = ("Index_OPNw_Eingriff", "index_opnw_eingriff", "Index_OP_Eingriff")
OPERATION_ALIASES = ("operation", "Operation", "eingriff")
ZUGANG_ALIASES = ("zugang", "Zugang")
_NULL_TOKENS = {"null", "nan", "none"}


def _clean(value) -> str:
    text = normalize_str(value)
    return "" if text.lower() in _NULL_TOKENS else text


def _optional_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    try:
        return _find_column(df, tuple(aliases), aliases[0])
    except ValueError:
        return None


def _is_reop_name(path: Path) -> bool:
    return "reop" in path.name.lower().replace("-", "").replace("_", "")


def looks_like_her_reop_table(path: Path) -> bool:
    """'REOP' in the file name + FallNummer + operation column.

    The name is required: the general ``HER_OP_Bericht*`` export shares the
    ``operation`` / ``zugang`` columns but lists ALL operations, so treating it
    as a Re-OP-only export would turn every unlisted case into "Nein".
    """
    if not _is_reop_name(path):
        return False
    try:
        df = read_table(path)
    except Exception:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    has_fall = any(a.lower() in cols for a in FALL_ALIASES)
    has_op = any(a.lower() in cols for a in OPERATION_ALIASES)
    return has_fall and has_op


def _format_reop_text(fall: str, sub: pd.DataFrame, cols: Dict[str, Optional[str]]) -> str:
    parts = [f"[ReOP-Bericht] FallNummer={fall} | Anzahl Folge-OPs={len(sub)}"]
    if cols["index"]:
        index_ops = [x for x in dict.fromkeys(sub[cols["index"]].map(_clean)) if x]
        if index_ops:
            parts.append("[Index_OPNw_Eingriff]\n" + "\n".join(index_ops))
    lines: List[str] = []
    for i, (_, row) in enumerate(sub.iterrows(), start=1):
        datum = _clean(row.get(cols["berdat"], ""))[:10] if cols["berdat"] else ""
        op = _clean(row.get(cols["operation"], "")) or "(kein OP-Titel)"
        zugang = _clean(row.get(cols["zugang"], "")) if cols["zugang"] else ""
        line = f"{i}. {datum}: {op}" if datum else f"{i}. {op}"
        if zugang:
            line += f" (Zugang: {zugang})"
        lines.append(line)
    parts.append("[operation]\n" + "\n".join(lines))
    return "\n\n".join(parts)


def build_reop_by_fall(df: pd.DataFrame) -> Dict[str, dict]:
    fall_col = _find_column(df, FALL_ALIASES, "FallNummer")
    cols = {
        "operation": _find_column(df, OPERATION_ALIASES, "operation"),
        "index": _optional_column(df, INDEX_OP_ALIASES),
        "zugang": _optional_column(df, ZUGANG_ALIASES),
        "berdat": _optional_column(df, BERDAT_ALIASES),
    }
    work = df.copy()
    work["_fall_nummer"] = work[fall_col].map(normalize_str)
    work = work[work["_fall_nummer"] != ""]
    if cols["berdat"]:
        work["_berdat_sort"] = pd.to_datetime(
            work[cols["berdat"]].map(_clean), errors="coerce"
        )
        work = work.sort_values(
            ["_fall_nummer", "_berdat_sort"], kind="stable", na_position="first"
        )
    else:
        LOGGER.warning("No berdat on Re-OP export; keeping file order per FallNummer.")

    out: Dict[str, dict] = {}
    for fall, sub in work.groupby("_fall_nummer", sort=False):
        out[str(fall)] = {
            REOP_TEXT_KEY: _format_reop_text(str(fall), sub, cols),
            "reop_fallnr": str(fall),
            "reop_n_ops": int(len(sub)),
        }
    return out


def load_reop_by_fall(paths: Sequence[PathLike]) -> Dict[str, dict]:
    frames: List[pd.DataFrame] = []
    resolved: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Re-OP input missing: {path}")
        frames.append(read_table(path))
        resolved.append(path)
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    by_fall = build_reop_by_fall(combined)
    LOGGER.info(
        "Loaded Re-OP reports for %d FallNummer(n) from %d file(s) (%d OP rows): %s",
        len(by_fall),
        len(resolved),
        len(combined),
        ", ".join(p.name for p in resolved),
    )
    return by_fall


def pick_reop_for_falls(by_fall: Dict[str, dict], fall_nummers: Sequence[str]) -> Optional[dict]:
    for fall in fall_nummers:
        key = normalize_str(fall)
        if key and key in by_fall:
            return dict(by_fall[key])
    return None


def discover_her_reop_paths(raw_dir: Optional[Path] = None) -> List[Path]:
    """``HER_*`` files under data/raw/ with 'reop' in the name (case-insensitive)."""
    from configs.config import RAW_DATA_DIR

    root = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    if not root.exists():
        return []
    suffixes = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm"}
    found = [
        p
        for p in root.glob("HER_*")
        if p.is_file()
        and _is_reop_name(p)
        and (p.suffix.lower() in suffixes or is_excel_path(p))
    ]
    by_stem: Dict[str, Path] = {}
    for path in sorted(found):
        prev = by_stem.get(path.stem)
        if prev is None or (prev.suffix.lower() != ".csv" and path.suffix.lower() == ".csv"):
            by_stem[path.stem] = path
    return sorted(by_stem.values(), key=lambda p: p.name.lower())
