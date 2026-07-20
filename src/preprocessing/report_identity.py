"""
Deterministic report identity helpers.

A stable ``source_row_id`` is assigned from load order so predictions can always
be merged back to their source rows regardless of later filtering. Task-agnostic.
"""

from __future__ import annotations

import pandas as pd

SOURCE_ROW_ID_COL = "source_row_id"


def normalize_str(value: object) -> str:
    """Return a stripped string; ``NaN``/``None``/``nan`` -> empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s


def assign_source_row_ids(df: pd.DataFrame, prefix: str = "row_") -> pd.DataFrame:
    """
    Assign ``source_row_id = <prefix><index>`` from row order.

    Call this before any row filtering so ids remain stable across the run.
    """
    out = df.reset_index(drop=True)
    out[SOURCE_ROW_ID_COL] = prefix + out.index.astype(str)
    return out
