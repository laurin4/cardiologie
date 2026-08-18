#!/usr/bin/env python3
"""
Diagnose Diagnoseliste ↔ Verlegungsbericht patient-ID join.

Reports overlap stats and sample IDs that fail to match. No LLM, no PHI upload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _ids_from_table(path: Path, aliases: tuple[str, ...], label: str) -> set[str]:
    df = read_table(path)
    col = _find_column(df, aliases, label)
    return {normalize_str(v) for v in df[col].tolist() if normalize_str(v)}


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

    diag_ids: set[str] = set()
    for p in diagnose_paths:
        ids = _ids_from_table(p, PATIENT_ID_ALIASES, "patient id")
        print(f"Diagnose {p.name}: {len(ids)} unique PatientID")
        diag_ids |= ids

    verl_ids: set[str] = set()
    for p in verlegung_paths:
        ids = _ids_from_table(p, PATIENT_ALIASES, "patnr")
        print(f"Verlegung {p.name}: {len(ids)} unique patnr")
        verl_ids |= ids

    overlap = diag_ids & verl_ids
    only_diag = diag_ids - verl_ids
    only_verl = verl_ids - diag_ids

    print()
    print(f"Diagnose patients total:   {len(diag_ids)}")
    print(f"Verlegung patients total:  {len(verl_ids)}")
    print(f"JOIN overlap (matched):    {len(overlap)}")
    print(f"Only in Diagnose:          {len(only_diag)}")
    print(f"Only in Verlegung:         {len(only_verl)}")
    if diag_ids:
        pct = 100.0 * len(overlap) / len(diag_ids)
        print(f"Match rate (of Diagnose):  {pct:.1f}%")

    n = max(0, args.sample)
    if n and only_diag:
        print()
        print(f"Sample Diagnose IDs with NO Verlegung ({min(n, len(only_diag))}):")
        for pid in sorted(only_diag)[:n]:
            print(f"  {pid}")
    if n and only_verl:
        print()
        print(f"Sample Verlegung IDs with NO Diagnose ({min(n, len(only_verl))}):")
        for pid in sorted(only_verl)[:n]:
            print(f"  {pid}")
    if n and overlap:
        print()
        print(f"Sample matched IDs ({min(n, len(overlap))}):")
        for pid in sorted(overlap)[:n]:
            print(f"  {pid}")

    if not overlap:
        print()
        print(
            "WARNING: 0 overlap — merge attaches empty verlegung_text for all Diagnose "
            "patients (no hard error). Check ID formats / whether files cover the same cohort."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
