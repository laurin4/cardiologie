from src.evaluation.metrics import (
    binary_confusion,
    binary_metrics_from_counts,
    categorical_accuracy,
    exact_match_rate,
)


def test_binary_metrics_from_counts():
    m = binary_metrics_from_counts(tp=8, fp=2, fn=2, tn=8)
    assert m["precision"] == 0.8
    assert m["recall"] == 0.8
    assert m["f1"] == 0.8
    assert m["accuracy"] == 0.8


def test_binary_confusion_truthy():
    y_true = [True, True, False, False]
    y_pred = ["yes", "no", "true", "false"]
    tp, fp, fn, tn = binary_confusion(y_true, y_pred)
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)


def test_categorical_accuracy():
    res = categorical_accuracy(["a", "b", "c"], ["a", "b", "x"])
    assert res["support"] == 3
    assert res["accuracy"] == round(2 / 3, 4)


def test_exact_match_rate():
    preds = [{"x": 1, "y": 2}, {"x": 1, "y": 9}]
    gt = [{"x": 1, "y": 2}, {"x": 1, "y": 2}]
    assert exact_match_rate(preds, gt, ["x", "y"]) == 0.5
