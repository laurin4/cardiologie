#!/usr/bin/env python3
"""
Diagnose Diagnoseliste ↔ Verlegungsbericht patient-ID join.

Reports overlap stats, ID format hints, and sample IDs that fail to match.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.diagnose_loader import (
    PATIENT_ID_ALIASES,
    _find_column,
    looks_like_her_diagnose_table,
)
from src.preprocessing.report_identity import normalize_str
from src.preprocessing.report_loader import discover_her_diagnose_paths
from src.preprocessing.verlegung_loader import (
    PATIENT_ALIASES,
    discover_her_verlegung_paths,
    looks_like_her_verlegung_table,
)
from src.utils.table_io import read_table


def _collect_ids(
    paths: Sequence[Path], aliases: tuple[str, ...], label: str
) -> Tuple[set[str], List[str], List[object], str, List[str]]:
    """Return (normalized_ids, raw_as_str_samples, raw_values, column_name, file_names)."""
    ids: set[str] = set()
    raw_samples: List[str] = []
    raw_values: List[object] = []
    col_used = ""
    files: List[str] = []
    for path in paths:
        df = read_table(path)
        col = _find_column(df, aliases, label)
        col_used = col
        files.append(path.name)
        series = df[col]
        print(f"  {path.name}: col={col!r} dtype={series.dtype} rows={len(series)}")
        for v in series.head(8).tolist():
            raw_values.append(v)
            raw_samples.append(repr(v))
        for v in series.tolist():
            nid = normalize_str(v)
            if nid:
                ids.add(nid)
    return ids, raw_samples, raw_values, col_used, files


def _id_shape(s: str) -> str:
    if re.fullmatch(r"\d+", s):
        return f"digits_len_{len(s)}"
    if re.fullmatch(r"\d+\.\d+", s):
        return "decimal_string"
    if re.fullmatch(r"[A-Za-z].*", s):
        return "starts_alpha"
    return f"other_len_{len(s)}"


def _print_format(name: str, ids: set[str], raw_samples: List[str]) -> None:
    print(f"\n=== Format: {name} ===")
    print(f"raw samples (repr): {raw_samples[:8]}")
    if not ids:
        print("normalized: (empty)")
        return
    shapes = Counter(_id_shape(x) for x in ids)
    lengths = Counter(len(x) for x in ids)
    print(f"normalized samples: {sorted(ids)[:8]}")
    print(f"shape counts: {dict(shapes.most_common(8))}")
    print(f"length counts: {dict(lengths.most_common(8))}")


def _strip_leading_zeros(ids: set[str]) -> set[str]:
    out = set()
    for x in ids:
        if x.isdigit():
            out.add(str(int(x)))
        else:
            out.add(x.lstrip("0") or "0")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Diagnoseliste↔Verlegung ID join")
    parser.add_argument(
        "--diagnose",
        nargs="*",
        default=None,
        help="HER Diagnose paths (default: all under data/raw/)",
    )
    parser.add_argument(
        "--verlegung",
        nargs="*",
        default=None,
        help="HER Verlegung paths (default: all under data/raw/)",
    )
    parser.add_argument("--sample", type=int, default=15, help="Sample unmatched IDs to print")
    args = parser.parse_args()

    diagnose_paths = (
        [Path(p) for p in args.diagnose]
        if args.diagnose
        else discover_her_diagnose_paths()
    )
    verlegung_paths = (
        [Path(p) for p in args.verlegung]
        if args.verlegung
        else discover_her_verlegung_paths()
    )

    if not diagnose_paths:
        raise SystemExit("No HER_Diagnose* files found under data/raw/.")
    if not verlegung_paths:
        raise SystemExit("No HER_Verlegungsbericht* files found under data/raw/.")

    for p in diagnose_paths:
        if not looks_like_her_diagnose_table(p):
            raise SystemExit(f"Not a Diagnoseliste table: {p}")
    for p in verlegung_paths:
        if not looks_like_her_verlegung_table(p):
            raise SystemExit(f"Not a Verlegung table: {p}")

    print("=== Files / columns ===")
    print("Diagnose:")
    diag_ids, diag_raw, _, _, _ = _collect_ids(
        diagnose_paths, PATIENT_ID_ALIASES, "patient id"
    )
    print("Verlegung:")
    verl_ids, verl_raw, _, _, _ = _collect_ids(
        verlegung_paths, PATIENT_ALIASES, "patnr"
    )

    _print_format("Diagnose PatientID", diag_ids, diag_raw)
    _print_format("Verlegung patnr", verl_ids, verl_raw)

    overlap = diag_ids & verl_ids
    only_diag = diag_ids - verl_ids
    only_verl = verl_ids - diag_ids

    # Heuristic: leading-zero mismatch
    overlap_lstrip = _strip_leading_zeros(diag_ids) & _strip_leading_zeros(verl_ids)

    print("\n=== Join stats ===")
    print(f"Diagnose patients total:   {len(diag_ids)}")
    print(f"Verlegung patients total:  {len(verl_ids)}")
    print(f"JOIN overlap (matched):    {len(overlap)}")
    print(f"Only in Diagnose:          {len(only_diag)}")
    print(f"Only in Verlegung:         {len(only_verl)}")
    if diag_ids:
        print(f"Match rate (of Diagnose):  {100.0 * len(overlap) / len(diag_ids):.1f}%")
    print(f"Overlap if strip leading 0: {len(overlap_lstrip)}")

    n = max(0, args.sample)
    if n and only_diag:
        print(f"\nSample Diagnose IDs with NO Verlegung ({min(n, len(only_diag))}):")
        for pid in sorted(only_diag)[:n]:
            print(f"  {pid}")
    if n and only_verl:
        print(f"\nSample Verlegung IDs with NO Diagnose ({min(n, len(only_verl))}):")
        for pid in sorted(only_verl)[:n]:
            print(f"  {pid}")
    if n and overlap:
        print(f"\nSample matched IDs ({min(n, len(overlap))}):")
        for pid in sorted(overlap)[:n]:
            print(f"  {pid}")

    if not overlap:
        print(
            "\nWARNING: 0 overlap — Verlegung is NOT merged onto Diagnose patients "
            "(empty verlegung_text; no hard pipeline error)."
        )
        if overlap_lstrip:
            print(
                "HINT: stripping leading zeros would create overlap — ID padding mismatch."
            )
        else:
            print(
                "HINT: likely different cohorts OR different ID spaces "
                "(PatientID vs patnr not the same key). Compare sample IDs above; "
                "ask Jasmin/Rodney which key links the two exports."
            )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
