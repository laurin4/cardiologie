#!/usr/bin/env python3
"""Export a review sheet (CSV by default) for manual consistency checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import PREDICTIONS_DIR
from configs.tasks import available_tasks, load_task
from src.evaluation.review_sheet import (
    build_review_rows,
    enrich_fall_keys_from_raw,
    load_result_rows,
    write_csv,
    write_excel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export review sheet for manual labeling")
    parser.add_argument("--task", default="cardiology_smoke")
    parser.add_argument(
        "--results",
        default=None,
        help="Path to predictions CSV (default: outputs/extractions/<task>_results.csv)",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "xlsx", "both"),
        default="csv",
        help="Output format (default: csv — opens reliably on USZ Windows)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path without forcing extension (default: outputs/evaluation/<task>_review.csv)",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of patients")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed when --max-rows is set")
    parser.add_argument(
        "--no-enrich-fall",
        action="store_true",
        help="Do not look up FallNummer from data/raw/ HER files",
    )
    parser.add_argument(
        "--only-verlegung-matched",
        action="store_true",
        help="After enrich, keep only rows with verlegung_matched=True",
    )
    args = parser.parse_args()

    try:
        load_task(args.task)
    except KeyError as exc:
        raise SystemExit(f"{exc}. Available: {available_tasks()}") from exc

    results_path = (
        Path(args.results)
        if args.results
        else PREDICTIONS_DIR / f"{args.task}_results.csv"
    )
    if not results_path.exists():
        raise SystemExit(f"Results not found: {results_path}")

    rows = load_result_rows(results_path)
    if not args.no_enrich_fall:
        rows = enrich_fall_keys_from_raw(rows)

    if args.only_verlegung_matched:
        rows = [
            r
            for r in rows
            if str(r.get("verlegung_matched", "")).strip().lower() in ("true", "1", "yes")
        ]
        print(f"Filtered to {len(rows)} Verlegung-matched rows")
        if not rows:
            raise SystemExit(
                "No Verlegung-matched patients in this results file. "
                "Re-run pipeline with: --require-verlegung --max-reports 15"
            )

    if args.max_rows is not None and args.max_rows > 0 and len(rows) > args.max_rows:
        import random

        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.max_rows)

    review_rows = build_review_rows(rows)
    base = Path(args.out) if args.out else Path("outputs/evaluation") / f"{args.task}_review"

    written: list[Path] = []
    if args.format in ("csv", "both"):
        csv_path = base if base.suffix.lower() == ".csv" else base.with_suffix(".csv")
        write_csv(review_rows, csv_path)
        written.append(csv_path)
    if args.format in ("xlsx", "both"):
        xlsx_path = base if base.suffix.lower() == ".xlsx" else base.with_suffix(".xlsx")
        write_excel(review_rows, xlsx_path)
        written.append(xlsx_path)

    n_matched = sum(1 for r in review_rows if r.get("verlegung_matched") == "True")
    print(f"Wrote {len(review_rows)} rows -> {', '.join(str(p) for p in written)}")
    if not args.no_enrich_fall:
        print(
            f"FallNummer enriched from data/raw/; "
            f"Verlegung matched: {n_matched}/{len(review_rows)}"
        )


if __name__ == "__main__":
    main()
