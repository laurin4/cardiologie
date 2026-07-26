"""Tests for retry-failed merge helpers."""

from src.pipeline.records_io import failed_keys, merge_rows_by_key, resume_key


def test_failed_keys():
    rows = [
        {"report_id": "a", "status": "extracted"},
        {"report_id": "b", "status": "failed"},
        {"source_row_id": "patient_2", "report_id": "", "status": "failed"},
    ]
    keys = failed_keys(rows)
    assert ("report_id", "b") in keys
    assert ("source_row_id", "patient_2") in keys
    assert ("report_id", "a") not in keys


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
