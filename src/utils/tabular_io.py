"""
Robust tabular CSV reading with encoding and separator fallbacks.

Task-agnostic. Handles the common clinical-export cases (semicolon-separated,
mixed encodings) without failing on individual malformed rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_ENCODINGS = ["utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"]
_SEPARATORS = [",", ";", "\t"]


def read_tabular(path, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Read a delimited text file into a dataframe, trying several encodings.

    If *sep* is None, the separator is auto-detected by picking the candidate that
    yields the most columns. Malformed rows are skipped rather than fatal.
    """
    path = Path(path)
    separators = [sep] if sep is not None else _SEPARATORS

    best: Optional[pd.DataFrame] = None
    best_cols = -1
    for enc in _ENCODINGS:
        for candidate in separators:
            try:
                df = pd.read_csv(
                    path,
                    sep=candidate,
                    encoding=enc,
                    engine="python",
                    dtype=str,
                    on_bad_lines="skip",
                )
            except Exception:
                continue
            n_cols = df.shape[1]
            if n_cols > best_cols:
                best = df
                best_cols = n_cols
        if best is not None and sep is not None:
            break

    if best is None:
        raise ValueError(f"CSV file could not be read: {path}")

    best.columns = [str(c).strip() for c in best.columns]
    return best
