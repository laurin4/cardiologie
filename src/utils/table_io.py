"""
Read CSV or Excel tables into a dataframe.

Sensitive clinical exports are usually semicolon-separated CSV on the server;
``.xlsx`` is still accepted as a fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.tabular_io import read_tabular

_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}


def read_excel_table(path: Path) -> pd.DataFrame:
    """Load the first sheet of an Excel workbook as string-typed columns."""
    path = Path(path)
    try:
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "Reading Excel requires openpyxl. Install with: pip install openpyxl"
        ) from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_table(path, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Load a tabular file.

    - ``.csv`` / ``.tsv`` / ``.txt`` -> :func:`read_tabular`
    - ``.xlsx`` / ``.xls`` / ``.xlsm`` -> :func:`read_excel_table`
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _EXCEL_SUFFIXES:
        return read_excel_table(path)
    return read_tabular(path, sep=sep)


def is_excel_path(path: Path) -> bool:
    return Path(path).suffix.lower() in _EXCEL_SUFFIXES
