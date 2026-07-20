from configs.tasks.base import ExtractionTask, SchemaField
from src.validation.schema_validation import validate_against_schema

TASK = ExtractionTask(
    name="t",
    fields=(
        SchemaField("flag", "boolean", required=True, default=False),
        SchemaField("grade", "enum", enum=("none", "mild", "severe"), default="none"),
        SchemaField("count", "integer", default=None),
        SchemaField("score", "number", default=None),
        SchemaField("note", "string", default=""),
    ),
)


def test_valid_payload():
    out, errors = validate_against_schema(
        {"flag": True, "grade": "mild", "count": "3", "score": "1.5", "note": " hi "},
        TASK,
    )
    assert out == {"flag": True, "grade": "mild", "count": 3, "score": 1.5, "note": "hi"}
    assert errors == []


def test_missing_required_reports_error_and_default():
    out, errors = validate_against_schema({}, TASK)
    assert out["flag"] is False
    assert any("flag" in e for e in errors)


def test_enum_case_insensitive_and_invalid():
    out, errors = validate_against_schema({"flag": False, "grade": "SEVERE"}, TASK)
    assert out["grade"] == "severe"
    out2, errors2 = validate_against_schema({"flag": False, "grade": "extreme"}, TASK)
    assert out2["grade"] == "none"
    assert any("grade" in e for e in errors2)


def test_boolean_coercion_from_strings():
    out, _ = validate_against_schema({"flag": "yes", "grade": "none"}, TASK)
    assert out["flag"] is True


def test_unknown_keys_ignored():
    out, _ = validate_against_schema({"flag": True, "grade": "none", "extra": 1}, TASK)
    assert "extra" not in out
