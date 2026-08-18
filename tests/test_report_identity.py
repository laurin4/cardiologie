"""Tests for patient-id normalization used in Diagnose↔Verlegung joins."""

from src.preprocessing.report_identity import normalize_str


def test_normalize_strips_excel_float_ids():
    assert normalize_str(12345.0) == "12345"
    assert normalize_str("12345.0") == "12345"
    assert normalize_str("12345.00") == "12345"
    assert normalize_str(" 12345 ") == "12345"


def test_normalize_keeps_non_int_floats_and_empty():
    assert normalize_str(12.5) == "12.5"
    assert normalize_str(None) == ""
    assert normalize_str(float("nan")) == ""
