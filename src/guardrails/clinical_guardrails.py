"""
Generic clinical guardrails / consistency checks.

These guardrails do NOT make a diagnosis. They are deterministic, task-declared
validation and normalization steps applied after schema validation:

- normalize        : map raw values to canonical ones
- conditional_set  : force a dependent field value when a condition holds
- mutually_exclusive : at most one of a set of boolean fields may be true
- requires         : if a condition holds, listed fields must be populated
- impossible_combination : flag value combinations that cannot co-occur

Rules are plain dicts on ``task.consistency_rules`` so new tasks add checks
without touching this engine. The engine returns the (possibly normalized) fields
plus a report of any issues and whether manual review is warranted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from configs.tasks.base import ExtractionTask


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    return str(a).strip().lower() == str(b).strip().lower()


def _condition_holds(fields: Dict[str, Any], condition: Dict[str, Any]) -> bool:
    """All key==value pairs in *condition* must match *fields*."""
    return all(_eq(fields.get(k), v) for k, v in condition.items())


def _apply_normalize(fields: Dict[str, Any], rule: Dict[str, Any], report: Dict[str, Any]) -> None:
    field = rule.get("field")
    mapping = rule.get("map", {})
    if field not in fields or _is_empty(fields.get(field)):
        return
    current = str(fields[field]).strip()
    lower_map = {str(k).strip().lower(): v for k, v in mapping.items()}
    if current.lower() in lower_map:
        new_value = lower_map[current.lower()]
        if not _eq(new_value, current):
            fields[field] = new_value
            report["normalized_fields"].append(f"{field}: {current!r} -> {new_value!r}")


def _apply_conditional_set(fields: Dict[str, Any], rule: Dict[str, Any], report: Dict[str, Any]) -> None:
    if _condition_holds(fields, rule.get("if", {})):
        for field, value in rule.get("set", {}).items():
            if not _eq(fields.get(field), value):
                report["normalized_fields"].append(
                    f"{field}: {fields.get(field)!r} -> {value!r} (conditional_set)"
                )
                fields[field] = value


def _apply_mutually_exclusive(fields: Dict[str, Any], rule: Dict[str, Any], report: Dict[str, Any]) -> None:
    group = rule.get("fields", [])
    truthy = [f for f in group if bool(fields.get(f))]
    if len(truthy) > 1:
        report["contradictions"].append(
            f"mutually_exclusive violated: {truthy} are all true"
        )


def _apply_requires(fields: Dict[str, Any], rule: Dict[str, Any], report: Dict[str, Any]) -> None:
    if _condition_holds(fields, rule.get("if", {})):
        for field in rule.get("then_required", []):
            if _is_empty(fields.get(field)):
                report["missing_dependent_fields"].append(
                    f"'{field}' required when {rule.get('if')}"
                )


def _apply_impossible_combination(
    fields: Dict[str, Any], rule: Dict[str, Any], report: Dict[str, Any]
) -> None:
    when = rule.get("when", {})
    if when and _condition_holds(fields, when):
        report["contradictions"].append(f"impossible_combination: {when}")


# Transformation rules mutate the output; detection rules only read the
# original model output so contradictions reflect what the LLM actually returned.
_TRANSFORM_RULES = {
    "normalize": _apply_normalize,
    "conditional_set": _apply_conditional_set,
}
_DETECTION_RULES = {
    "mutually_exclusive": _apply_mutually_exclusive,
    "requires": _apply_requires,
    "impossible_combination": _apply_impossible_combination,
}


def apply_guardrails(
    fields: Dict[str, Any],
    task: ExtractionTask,
    evidence_bundle: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply the task's consistency rules to validated *fields*.

    Detection rules (``impossible_combination``, ``mutually_exclusive``,
    ``requires``) are evaluated against the original model output. Transformation
    rules (``normalize``, ``conditional_set``) produce the final fields.

    Returns ``(final_fields, report)``. ``report`` contains:
        normalized_fields, contradictions, missing_dependent_fields,
        manual_review_candidate (bool), applied_rules (list of rule types).
    """
    original = dict(fields)
    out = dict(fields)
    report: Dict[str, Any] = {
        "normalized_fields": [],
        "contradictions": [],
        "missing_dependent_fields": [],
        "applied_rules": [],
    }

    for rule in task.consistency_rules:
        rtype = str(rule.get("type", ""))
        if rtype in _TRANSFORM_RULES:
            _TRANSFORM_RULES[rtype](out, rule, report)
        elif rtype in _DETECTION_RULES:
            _DETECTION_RULES[rtype](original, rule, report)
        else:
            report["contradictions"].append(f"unknown_rule_type: {rtype}")
            continue
        report["applied_rules"].append(rtype)

    report["manual_review_candidate"] = bool(
        report["contradictions"] or report["missing_dependent_fields"]
    )
    return out, report
