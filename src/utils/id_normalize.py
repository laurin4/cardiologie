"""
Generic identifier / dataframe helpers shared across the framework.

Task-agnostic: no clinical or task-specific column names are hard-coded here.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when a required column is missing from a dataframe."""


def clean_id_value(value: object) -> str:
    """
    Normalize a single identifier to a stripped string without float artifacts.

    Examples: ``12345 -> "12345"``, ``12345.0 -> "12345"``, ``NaN -> ""``.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        try:
            if value == int(value):
                return str(int(value))
        except (TypeError, ValueError, OverflowError):
            pass
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit() or (head.startswith("-") and head[1:].isdigit()):
            return head
    return s


def normalize_id_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy with *column* cleaned via :func:`clean_id_value` (merge-safe)."""
    out = df.copy()
    if column in out.columns:
        out[column] = out[column].map(clean_id_value)
    return out


def require_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    context: str,
) -> None:
    """Raise :class:`SchemaValidationError` listing missing and available columns."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        available = list(df.columns)
        raise SchemaValidationError(
            f"{context}: missing required column(s): {', '.join(missing)}. "
            f"Available columns: {available}"
        )
