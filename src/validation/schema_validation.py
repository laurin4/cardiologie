"""
Schema validation for LLM extraction output.

Coerces a raw LLM JSON object into the task's declared schema: enforces required
fields, coerces types, normalizes enum values (case-insensitive), fills defaults,
and drops unknown keys. Returns the validated object plus a list of human-readable
issues (never raises on bad model output).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from configs.tasks.base import ExtractionTask, SchemaField

_TRUE_STRINGS = {"1", "true", "yes", "y", "ja", "wahr"}
_FALSE_STRINGS = {"0", "false", "no", "n", "nein", "falsch"}


def _coerce_boolean(value: Any) -> Tuple[Any, bool]:
    if isinstance(value, bool):
        return value, True
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value), True
    s = str(value).strip().lower()
    if s in _TRUE_STRINGS:
        return True, True
    if s in _FALSE_STRINGS:
        return False, True
    return None, False


def _coerce_scalar(field: SchemaField, value: Any) -> Tuple[Any, bool]:
    """Return (coerced_value, ok)."""
    t = field.type
    if t == "string":
        return str(value).strip(), True
    if t == "boolean":
        return _coerce_boolean(value)
    if t == "integer":
        try:
            return int(float(str(value).strip())), True
        except (TypeError, ValueError):
            return None, False
    if t == "number":
        try:
            return float(str(value).strip()), True
        except (TypeError, ValueError):
            return None, False
    if t == "enum":
        s = str(value).strip()
        lower_map = {e.lower(): e for e in field.enum}
        if s.lower() in lower_map:
            return lower_map[s.lower()], True
        return None, False
    if t == "array":
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()], True
        s = str(value).strip()
        return ([s] if s else []), True
    return value, True


def validate_against_schema(
    raw: Dict[str, Any], task: ExtractionTask
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate/coerce *raw* against *task* schema.

    Returns ``(validated_dict, errors)``. ``validated_dict`` always contains every
    schema field (using defaults where needed). ``errors`` lists all problems.
    """
    raw = raw if isinstance(raw, dict) else {}
    validated: Dict[str, Any] = {}
    errors: List[str] = []

    for field in task.fields:
        present = field.name in raw and raw[field.name] is not None
        if not present:
            if field.required:
                errors.append(f"missing required field '{field.name}'")
            validated[field.name] = field.default
            continue

        value = raw[field.name]
        coerced, ok = _coerce_scalar(field, value)
        if not ok:
            detail = f"invalid value for '{field.name}' (type={field.type})"
            if field.type == "enum":
                detail += f"; got {value!r}, allowed {list(field.enum)}"
            errors.append(detail)
            validated[field.name] = field.default
        else:
            validated[field.name] = coerced

    return validated, errors
