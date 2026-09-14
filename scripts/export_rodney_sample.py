#!/usr/bin/env python3
"""Export Rodney review Excel: N patients × all variables (same cohort)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import PREDICTIONS_DIR, RAW_DATA_DIR
from src.evaluation.dendrite_score import discover_dendrite_paths, load_dendrite_fall_ids
from src.evaluation.review_sheet import CLINICAL_COLUMNS, enrich_fall_keys_from_raw, load_result_rows
from src.evaluation.rodney_review import (
    build_long_rows,
    filter_dendrite_overlap,
    sample_patients,
    write_rodney_csv,
    write_rodney_excel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export Rodney review for exactly N Dendrite-overlap patients; "
            "all variables for the same patients (equal counts)."
        )
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Predictions CSV (default: outputs/extractions_dendrite/cardiology_smoke_results.csv)",
    )
    parser.add_argument(
        "--dendrite",
        default=None,
        help="Dendrite gold file (default: discover under data/raw/)",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=25,
        help="Number of patients to include (default: 25). Same set for every variable.",
    )
    parser.add_argument(
        "--n-per-var",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # legacy alias; ignored — use --n-patients
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        choices=("xlsx", "csv", "both"),
        default="xlsx",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path base (default: outputs/evaluation/rodney_review_25)",
    )
    parser.add_argument(
        "--no-enrich-fall",
        action="store_true",
        help="Skip FallNummer enrichment from HER raw tables",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help=f"Subset of variables (default: all clinical). Options: {', '.join(CLINICAL_COLUMNS)}",
    )
    args = parser.parse_args()

    n_patients = args.n_patients
    if args.n_per_var is not None:
        print(
            "NOTE: --n-per-var is deprecated; sampling is by patient. "
            f"Using --n-patients {n_patients}."
        )

    results_path = (
        Path(args.results)
        if args.results
        else Path("outputs/extractions_dendrite/cardiology_smoke_results.csv")
    )
    if not results_path.exists():
        alt = PREDICTIONS_DIR / "cardiology_smoke_results.csv"
        if alt.exists():
            results_path = alt
        else:
            raise SystemExit(f"Results not found: {results_path}")

    if args.dendrite:
        dend_path = Path(args.dendrite)
    else:
        found = discover_dendrite_paths(RAW_DATA_DIR)
        if not found:
            raise SystemExit(
                f"No Dendrite* under {RAW_DATA_DIR}. Pass --dendrite explicitly."
            )
        dend_path = found[0]

    rows = load_result_rows(results_path)
    if not args.no_enrich_fall:
        rows = enrich_fall_keys_from_raw(rows)

    fall_ids = load_dendrite_fall_ids(dend_path)
    overlapped = filter_dendrite_overlap(rows, fall_ids)
    print(
        f"Dendrite overlap: {len(overlapped)}/{len(rows)} result rows "
        f"(Dendrite FallNummers={len(fall_ids)} from {dend_path.name})"
    )
    if not overlapped:
        raise SystemExit(
            "No Dendrite-overlap rows. Re-run pipeline with --dendrite and IPS Verlegung primary."
        )

    patients = sample_patients(overlapped, n_patients=n_patients, seed=args.seed)
    print(f"Selected {len(patients)} patients (seed={args.seed})")

    variables = args.variables or list(CLINICAL_COLUMNS)
    long_rows = build_long_rows(patients, variables=variables)
    counts = {}
    for r in long_rows:
        counts[r["variable"]] = counts.get(r["variable"], 0) + 1
    print(
        "Rows per variable (must be equal):",
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    )
    if len(set(counts.values())) > 1:
        raise SystemExit("Internal error: unequal variable row counts after patient sampling.")

    base = Path(args.out) if args.out else Path("outputs/evaluation") / f"rodney_review_{n_patients}"
    written: list[Path] = []
    if args.format in ("csv", "both"):
        csv_path = base if base.suffix.lower() == ".csv" else base.with_suffix(".csv")
        write_rodney_csv(long_rows, csv_path)
        written.append(csv_path)
    if args.format in ("xlsx", "both"):
        xlsx_path = base if base.suffix.lower() == ".xlsx" else base.with_suffix(".xlsx")
        write_rodney_excel(long_rows, xlsx_path)
        written.append(xlsx_path)

    print(f"Wrote {len(long_rows)} review rows -> {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
