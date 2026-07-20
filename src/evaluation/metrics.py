"""
Generic classification metrics.

Pure-Python helpers (no heavy dependency required) for scoring extracted fields
against ground truth. Works for boolean fields (binary metrics) and categorical /
enum fields (accuracy + per-class confusion).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def binary_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Precision/recall/F1/accuracy/specificity from confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "specificity": round(specificity, 4),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def binary_confusion(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> Tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` treating truthy values as the positive class."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        tb, pb = _as_bool(t), _as_bool(p)
        if tb and pb:
            tp += 1
        elif not tb and pb:
            fp += 1
        elif tb and not pb:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def binary_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, float]:
    """Binary metrics for two aligned label sequences."""
    return binary_metrics_from_counts(*binary_confusion(y_true, y_pred))


def categorical_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, Any]:
    """Accuracy plus a per-class confusion table for categorical/enum fields."""
    n = 0
    correct = 0
    confusion: Dict[str, Dict[str, int]] = {}
    for t, p in zip(y_true, y_pred):
        ts = "" if t is None else str(t).strip()
        ps = "" if p is None else str(p).strip()
        n += 1
        if ts == ps:
            correct += 1
        confusion.setdefault(ts, {})
        confusion[ts][ps] = confusion[ts].get(ps, 0) + 1
    return {
        "support": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "confusion": confusion,
    }


def exact_match_rate(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    fields: Sequence[str],
) -> float:
    """Fraction of records where all *fields* match exactly (aligned by order)."""
    if not predictions:
        return 0.0
    matches = 0
    for pred, gt in zip(predictions, ground_truth):
        if all(str(pred.get(f)) == str(gt.get(f)) for f in fields):
            matches += 1
    return round(matches / len(predictions), 4)
