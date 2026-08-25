#!/usr/bin/env python3
"""
Diagnose Diagnoseliste ↔ Verlegungsbericht FallNummer join.

Reports overlap stats, ID format hints, and sample IDs that fail to match.
Clinic join key: FallNummer (not PatientID/patnr).
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
    FALL_ALIASES as DIAG_FALL_ALIASES,
    PATIENT_ID_ALIASES,
    _find_column,
    looks_like_her_diagnose_table,
)
from src.preprocessing.report_identity import normalize_str
from src.preprocessing.report_loader import discover_her_diagnose_paths
from src.preprocessing.verlegung_loader import (
    FALL_ALIASES as VERL_FALL_ALIASES,
    PATIENT_ALIASES,
    discover_her_verlegung_paths,
    looks_like_her_verlegung_table,
)
from src.utils.table_io import read_table


def _collect_ids(
    paths: Sequence[Path], aliases: tuple[str, ...], label: str
) -> Tuple[set[str], List[str], str]:
    ids: set[str] = set()
    raw_samples: List[str] = []
    col_used = ""
    for path in paths:
        df = read_table(path)
        col = _find_column(df, aliases, label)
        col_used = col
        series = df[col]
        print(f"  {path.name}: col={col!r} dtype={series.dtype} rows={len(series)}")
        for v in series.head(8).tolist():
            raw_samples.append(repr(v))
        for v in series.tolist():
            nid = normalize_str(v)
            if nid:
                ids.add(nid)
    return ids, raw_samples, col_used


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


def _print_join(name: str, left: set[str], right: set[str], sample: int) -> int:
    overlap = left & right
    only_left = left - right
    only_right = right - left
    print(f"\n=== Join stats ({name}) ===")
    print(f"Diagnose unique:           {len(left)}")
    print(f"Verlegung unique:          {len(right)}")
    print(f"JOIN overlap (matched):    {len(overlap)}")
    print(f"Only in Diagnose:          {len(only_left)}")
    print(f"Only in Verlegung:         {len(only_right)}")
    if left:
        print(f"Match rate (of Diagnose):  {100.0 * len(overlap) / len(left):.1f}%")
    print(
        f"Overlap if strip leading 0: "
        f"{len(_strip_leading_zeros(left) & _strip_leading_zeros(right))}"
    )
    n = max(0, sample)
    if n and only_left:
        print(f"\nSample Diagnose-only ({min(n, len(only_left))}):")
        for x in sorted(only_left)[:n]:
            print(f"  {x}")
    if n and only_right:
        print(f"\nSample Verlegung-only ({min(n, len(only_right))}):")
        for x in sorted(only_right)[:n]:
            print(f"  {x}")
    if n and overlap:
        print(f"\nSample matched ({min(n, len(overlap))}):")
        for x in sorted(overlap)[:n]:
            print(f"  {x}")
    return len(overlap)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Diagnoseliste↔Verlegung FallNummer join"
    )
    parser.add_argument("--diagnose", nargs="*", default=None)
    parser.add_argument("--verlegung", nargs="*", default=None)
    parser.add_argument("--sample", type=int, default=15)
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

    print("=== Primary join key: FallNummer ===")
    print("Diagnose FallNummer:")
    diag_falls, diag_fall_raw, _ = _collect_ids(
        diagnose_paths, DIAG_FALL_ALIASES, "FallNummer"
    )
    print("Verlegung FallNummer/fallnr:")
    verl_falls, verl_fall_raw, _ = _collect_ids(
        verlegung_paths, VERL_FALL_ALIASES, "FallNummer"
    )
    _print_format("Diagnose FallNummer", diag_falls, diag_fall_raw)
    _print_format("Verlegung FallNummer", verl_falls, verl_fall_raw)
    fall_overlap = _print_join("FallNummer", diag_falls, verl_falls, args.sample)

    print("\n=== Reference only (not used for merge): PatientID vs patnr ===")
    print("Diagnose PatientID:")
    diag_ids, diag_raw, _ = _collect_ids(
        diagnose_paths, PATIENT_ID_ALIASES, "patient id"
    )
    print("Verlegung patnr:")
    verl_ids, verl_raw, _ = _collect_ids(verlegung_paths, PATIENT_ALIASES, "patnr")
    _print_format("Diagnose PatientID", diag_ids, diag_raw)
    _print_format("Verlegung patnr", verl_ids, verl_raw)
    _print_join("PatientID↔patnr (legacy)", diag_ids, verl_ids, min(5, args.sample))

    if not fall_overlap:
        print(
            "\nWARNING: 0 FallNummer overlap — Verlegung will NOT attach "
            "(empty verlegung_text). Likely cohort mismatch "
            "(e.g. Verlegung only 2025/2026 vs older Diagnose cases)."
        )
        raise SystemExit(2)

    print(f"\nOK: FallNummer overlap = {fall_overlap}")


if __name__ == "__main__":
    main()
