"""Tests for retry-failed merge helpers."""

from src.pipeline.records_io import failed_keys, merge_rows_by_key, resume_key
from src.preprocessing.report_loader import REPORT_ID_KEY


def test_failed_keys():
    rows = [
        {"report_id": "a", "status": "extracted"},
        {"report_id": "b", "status": "failed"},
        {"report_id": "c", "status": "partial"},
        {"source_row_id": "patient_2", "report_id": "", "status": "failed"},
    ]
    keys = failed_keys(rows)
    assert ("report_id", "b") in keys
    assert ("report_id", "c") in keys
    assert ("source_row_id", "patient_2") in keys
    assert ("report_id", "a") not in keys


def test_retry_todo_only_failed_ids_not_whole_dataset():
    """Regression: retry must not enqueue every patient missing from OK set."""
    all_existing = [
        {"report_id": "101", "source_row_id": "patient_0", "status": "extracted"},
        {"report_id": "102", "source_row_id": "patient_1", "status": "failed"},
    ]
    reports = [
        {REPORT_ID_KEY: "101", "source_row_id": "patient_0"},
        {REPORT_ID_KEY: "102", "source_row_id": "patient_1"},
        {REPORT_ID_KEY: "999", "source_row_id": "patient_700"},
    ]
    failed_key_set = failed_keys(all_existing)
    failed_report_ids = {"102"}
    todo = [
        r
        for r in reports
        if resume_key(r) in failed_key_set
        or str(r.get(REPORT_ID_KEY, "")).strip() in failed_report_ids
    ]
    assert len(todo) == 1
    assert todo[0][REPORT_ID_KEY] == "102"


def test_merge_rows_replaces_failed():
    existing = [
        {"report_id": "a", "status": "extracted", "v": 1},
        {"report_id": "b", "status": "failed", "v": 0},
    ]
    updates = [{"report_id": "b", "status": "extracted", "v": 9}]
    merged = merge_rows_by_key(existing, updates)
    assert len(merged) == 2
    assert merged[0]["v"] == 1
    assert merged[1]["status"] == "extracted"
    assert merged[1]["v"] == 9


def test_resume_key_prefers_source_row_id():
    assert resume_key({"source_row_id": "patient_0", "report_id": "x"}) == (
        "source_row_id",
        "patient_0",
    )
