#!/usr/bin/env python3
"""Score cardiology_smoke predictions against Dendrite postop gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import OUTPUTS_DIR, PREDICTIONS_DIR, RAW_DATA_DIR
from src.evaluation.dendrite_score import (
    discover_dendrite_paths,
    format_score_report,
    run_dendrite_score,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pipeline results vs Dendrite postop gold (FallNummer join)."
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Pipeline results CSV (default: outputs/extractions/cardiology_smoke_results.csv).",
    )
    parser.add_argument(
        "--dendrite",
        default=None,
        help="Dendrite gold Excel/CSV (default: discover Dendrite* under data/raw/).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write score JSON/CSV (default: outputs/evaluation/).",
    )
    args = parser.parse_args()

    pred_path = (
        Path(args.predictions)
        if args.predictions
        else PREDICTIONS_DIR / "cardiology_smoke_results.csv"
    )
    if not pred_path.exists():
        raise SystemExit(f"Predictions not found: {pred_path}")

    if args.dendrite:
        dend_path = Path(args.dendrite)
    else:
        found = discover_dendrite_paths(RAW_DATA_DIR)
        if not found:
            raise SystemExit(
                f"No Dendrite* file under {RAW_DATA_DIR}. Pass --dendrite explicitly."
            )
        dend_path = found[0]
        if len(found) > 1:
            print(f"Multiple Dendrite files; using {dend_path.name}")

    if not dend_path.exists():
        raise SystemExit(f"Dendrite file not found: {dend_path}")

    result = run_dendrite_score(pred_path, dend_path)
    report = format_score_report(result)
    print(report)

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "predictions": str(pred_path),
        "dendrite": str(dend_path),
        "n_aligned_fall": result["n_aligned_fall"],
        "per_field": result["per_field"],
    }
    json_path = out_dir / "dendrite_score.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pair_rows = []
    for field, rows in (result.get("pairs") or {}).items():
        pair_rows.extend(rows)
    pairs_path = out_dir / "dendrite_score_pairs.csv"
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(pairs_path, index=False, sep=";", encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=["fall", "field", "gold_raw", "gold", "pred_raw", "pred", "match"]
        ).to_csv(pairs_path, index=False, sep=";", encoding="utf-8-sig")

    print(f"Wrote {json_path}")
    print(f"Wrote {pairs_path}")
    if result["n_aligned_fall"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
