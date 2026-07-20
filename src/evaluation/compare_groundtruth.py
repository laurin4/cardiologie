"""
Compare extracted structured output against ground-truth annotations.

Aligns predictions and ground truth by a shared id key, then scores each schema
field with the appropriate metric (binary for boolean fields, accuracy + confusion
for enum/categorical fields). Task-agnostic: the field set comes from the task.
"""

from __future__ import annotations

from typing import Any, Dict, List

from configs.tasks.base import ExtractionTask
from src.evaluation.metrics import (
    binary_metrics,
    categorical_accuracy,
    exact_match_rate,
)


def align_by_id(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    id_key: str = "report_id",
) -> List[Dict[str, Any]]:
    """Return list of ``{"pred": ..., "gt": ...}`` for ids present in both."""
    gt_index = {str(g.get(id_key)): g for g in ground_truth}
    aligned: List[Dict[str, Any]] = []
    for pred in predictions:
        key = str(pred.get(id_key))
        if key in gt_index:
            aligned.append({"pred": pred, "gt": gt_index[key]})
    return aligned


def compare(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    task: ExtractionTask,
    id_key: str = "report_id",
) -> Dict[str, Any]:
    """
    Score *predictions* against *ground_truth* per task field.

    Returns ``{"n_aligned", "per_field": {field: metrics}, "exact_match_rate"}``.
    """
    aligned = align_by_id(predictions, ground_truth, id_key)
    preds = [a["pred"] for a in aligned]
    gts = [a["gt"] for a in aligned]

    per_field: Dict[str, Any] = {}
    for f in task.fields:
        y_pred = [p.get(f.name) for p in preds]
        y_true = [g.get(f.name) for g in gts]
        if f.type == "boolean":
            per_field[f.name] = {"type": "boolean", **binary_metrics(y_true, y_pred)}
        elif f.type in ("enum", "string"):
            per_field[f.name] = {"type": f.type, **categorical_accuracy(y_true, y_pred)}
        else:
            per_field[f.name] = {"type": f.type, **categorical_accuracy(y_true, y_pred)}

    return {
        "task": task.name,
        "n_aligned": len(aligned),
        "per_field": per_field,
        "exact_match_rate": exact_match_rate(
            preds, gts, [f.name for f in task.fields]
        ),
    }
