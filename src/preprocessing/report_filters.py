"""
Generic report metadata filters.

Task-agnostic helpers for normalizing an optional report-type column and excluding
report types a task does not want to process. Nothing here is task-specific;
the excluded types are supplied by the caller.
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

REPORT_TYPE_COLUMN = "report_type"


def normalize_report_type(value: object) -> str:
    """Return a stripped report-type string; ``NaN``/``None`` -> empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s


def exclude_report_types(
    df: pd.DataFrame,
    excluded_types: Iterable[str],
    *,
    column: str = REPORT_TYPE_COLUMN,
) -> Tuple[pd.DataFrame, int]:
    """
    Return a copy of *df* without rows whose *column* is in *excluded_types*.

    If the column is absent no rows are excluded (a warning is logged). Returns
    ``(filtered_df, excluded_count)``.
    """
    excluded = {str(t).strip() for t in excluded_types if str(t).strip()}
    if df.empty or not excluded:
        return df.copy(), 0
    if column not in df.columns:
        LOGGER.warning(
            "Report dataframe has no '%s' column; report-type exclusion skipped.", column
        )
        return df.copy(), 0
    mask = df[column].map(normalize_report_type).isin(excluded)
    n = int(mask.sum())
    if n:
        LOGGER.info("excluded_report_type_count=%d (%s)", n, sorted(excluded))
    return df.loc[~mask].copy(), n
