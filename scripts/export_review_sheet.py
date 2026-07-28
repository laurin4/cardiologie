#!/usr/bin/env python3
"""Export a review sheet (CSV by default) for manual consistency checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_task_config
from src.evaluation.review_sheet import (
    build_review_rows,
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
        help="Path to predictions CSV (default: outputs/results/<task>_results.csv)",
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
    args = parser.parse_args()

    cfg = load_task_config(args.task)
    results_path = Path(args.results) if args.results else Path(cfg["paths"]["results_csv"])
    if not results_path.exists():
        raise SystemExit(f"Results not found: {results_path}")

    rows = load_result_rows(results_path)
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

    print(f"Wrote {len(review_rows)} rows -> {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
