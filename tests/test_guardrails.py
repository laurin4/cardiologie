from configs.tasks.base import ExtractionTask, SchemaField
from src.guardrails.clinical_guardrails import apply_guardrails

TASK = ExtractionTask(
    name="t",
    fields=(
        SchemaField("present", "boolean", default=False),
        SchemaField("negated", "boolean", default=False),
        SchemaField("grade", "enum", enum=("none", "mild"), default="none"),
        SchemaField("a", "boolean", default=False),
        SchemaField("b", "boolean", default=False),
    ),
    consistency_rules=(
        {"type": "conditional_set", "if": {"negated": True}, "set": {"present": False}},
        {"type": "conditional_set", "if": {"present": False}, "set": {"grade": "none"}},
        {"type": "impossible_combination", "when": {"present": True, "negated": True}},
        {"type": "requires", "if": {"present": True}, "then_required": ["grade"]},
        {"type": "mutually_exclusive", "fields": ["a", "b"]},
    ),
)


def test_conditional_set_negation_forces_present_false():
    out, report = apply_guardrails({"present": True, "negated": True, "grade": "mild"}, TASK)
    assert out["present"] is False
    assert out["grade"] == "none"
    assert report["normalized_fields"]


def test_impossible_combination_flagged_before_normalization():
    # Contradiction is detected on the raw True/True input.
    _, report = apply_guardrails({"present": True, "negated": True, "grade": "mild"}, TASK)
    assert any("impossible_combination" in c for c in report["contradictions"])
    assert report["manual_review_candidate"] is True


def test_requires_missing_dependent_field():
    _, report = apply_guardrails({"present": True, "negated": False, "grade": ""}, TASK)
    assert report["missing_dependent_fields"]


def test_mutually_exclusive_violation():
    _, report = apply_guardrails({"present": False, "a": True, "b": True}, TASK)
    assert any("mutually_exclusive" in c for c in report["contradictions"])
