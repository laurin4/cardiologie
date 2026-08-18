"""
Deterministic report identity helpers.

A stable ``source_row_id`` is assigned from load order so predictions can always
be merged back to their source rows regardless of later filtering. Task-agnostic.
"""

from __future__ import annotations

import re

import pandas as pd

SOURCE_ROW_ID_COL = "source_row_id"

_FLOAT_INT_RE = re.compile(r"^\d+\.0+$")


def normalize_str(value: object) -> str:
    """
    Return a stripped string; ``NaN``/``None``/``nan`` -> empty string.

    Also normalizes Excel-style numeric IDs (``12345.0`` / float ``12345.0``)
    to ``12345`` so Diagnoseliste PatientID and Verlegung patnr can join.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    s = str(value).strip()
    if s.lower() in ("nan", "none"):
        return ""
    if _FLOAT_INT_RE.fullmatch(s):
        return s.split(".", 1)[0]
    return s


def assign_source_row_ids(df: pd.DataFrame, prefix: str = "row_") -> pd.DataFrame:
    """
    Assign ``source_row_id = <prefix><index>`` from row order.

    Call this before any row filtering so ids remain stable across the run.
    """
    out = df.reset_index(drop=True)
    out[SOURCE_ROW_ID_COL] = prefix + out.index.astype(str)
    return out
