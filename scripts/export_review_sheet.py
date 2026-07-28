#!/usr/bin/env python3
"""CLI: export a minimal Excel review sheet for manual consistency checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.config import PREDICTIONS_DIR
from src.evaluation.review_sheet import build_review_rows, load_result_rows, write_excel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a minimal Excel sheet for manual consistency review."
    )
    parser.add_argument("--task", default="cardiology_smoke")
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    csv_path = (
        Path(args.results_csv)
        if args.results_csv
        else PREDICTIONS_DIR / f"{args.task}_results.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    out_path = (
        Path(args.out)
        if args.out
        else Path("outputs/evaluation") / f"{args.task}_review.xlsx"
    )

    rows = build_review_rows(load_result_rows(csv_path))
    if args.max_rows is not None:
        rows = rows[: max(0, args.max_rows)]

    write_excel(rows, out_path)
    print(f"Wrote {len(rows)} review rows -> {out_path}")
    print("Fill: correct_swi / correct_reop / notes   (ja | nein | unsicher)")


if __name__ == "__main__":
    main()
