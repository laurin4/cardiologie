"""
LLM extraction stage.

Takes the rule-based evidence bundle plus a task specification and asks the LLM to
produce a structured JSON object matching the task schema. The raw model output is
parsed and validated against the schema before being returned.

Transient failures (empty response, timeout → empty string, non-JSON) are retried
a few times before degrading to schema defaults.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from configs.tasks.base import ExtractionTask
from src.llm.json_parsing import parse_llm_json_output
from src.llm.llm_debug import write_llm_debug
from src.llm.llm_interface import call_llm
from src.llm.model_config import LLM_MAX_RETRIES
from src.prompts.registry import build_system_prompt
from src.validation.schema_validation import validate_against_schema


def _default_fields(task: ExtractionTask) -> Dict[str, Any]:
    return {f.name: f.default for f in task.fields}


def _apply_normalize_maps(raw: Dict[str, Any], task: ExtractionTask) -> Dict[str, Any]:
    """Apply task normalize maps before schema validation (e.g. English -> German enums)."""
    out = dict(raw)
    for rule in task.consistency_rules:
        if str(rule.get("type", "")) != "normalize":
            continue
        field = rule.get("field")
        mapping = {str(k).strip().lower(): v for k, v in (rule.get("map") or {}).items()}
        if not field or field not in out or out[field] is None:
            continue
        key = str(out[field]).strip().lower()
        if key in mapping:
            out[field] = mapping[key]
    return out


def _retry_user_suffix(task: ExtractionTask, attempt: int) -> str:
    if attempt <= 0:
        return ""
    if (task.language or "en").lower().startswith("de"):
        return (
            "\n\nWICHTIG (Retry): Antworte AUSSCHLIESSLICH mit einem einzigen gültigen "
            "JSON-Objekt. Kein Markdown, kein erklärender Text ausserhalb des JSON."
        )
    return (
        "\n\nIMPORTANT (retry): Reply with ONLY a single valid JSON object. "
        "No markdown or prose outside the JSON."
    )


def extract_entities(
    evidence_bundle: Dict[str, Any],
    task: ExtractionTask,
    *,
    record_id: str = "",
    report_name: str = "",
    llm_text_override: str | None = None,
) -> Dict[str, Any]:
    """
    Run schema-guided LLM extraction over an evidence bundle.

    Returns a dict with validated fields, schema_errors, prompts, raw_output, etc.
    Never raises: on final failure it returns schema defaults with the error recorded.
    """
    llm_text = (
        llm_text_override
        if llm_text_override is not None
        else str(evidence_bundle.get("llm_report_text", "") or "")
    )
    system_prompt = build_system_prompt(task)
    if (task.language or "en").lower().startswith("de"):
        user_base = (
            "Klinische Evidenz zur Extraktion. Bevorzuge die gelabelten Snippets, "
            "falls vorhanden; andernfalls nutze die vollständige Diagnoseliste / "
            "den Berichtstext unten.\n\n"
            f"{llm_text}\n"
        )
    else:
        user_base = (
            "Clinical evidence for extraction. Prefer the labelled snippets when "
            "present; otherwise use the full Diagnoseliste / report text below.\n\n"
            f"{llm_text}\n"
        )

    max_attempts = max(1, 1 + max(0, LLM_MAX_RETRIES))
    raw_output = ""
    user_prompt = user_base
    last_error: Exception | None = None
    attempts_used = 0

    for attempt in range(max_attempts):
        attempts_used = attempt + 1
        user_prompt = user_base + _retry_user_suffix(task, attempt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw_output = call_llm(messages)
            parsed = parse_llm_json_output(raw_output, f"{task.name} / extraction")
            parsed = _apply_normalize_maps(parsed, task)
            validated, errors = validate_against_schema(parsed, task)
            return {
                "fields": validated,
                "schema_errors": errors,
                "raw_output": raw_output,
                "llm_called": True,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "reasoning": str(
                    validated.get("reasoning", "") or parsed.get("reasoning", "") or ""
                ),
                "evidence_quotes": validated.get(
                    "evidence_quotes", parsed.get("evidence_quotes", [])
                ),
                "attempts": attempts_used,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"LLM extraction attempt {attempts_used}/{max_attempts} "
                f"failed ({task.name}): {exc}"
            )
            if attempt + 1 < max_attempts:
                # Brief pause helps after timeouts / overloaded local API.
                time.sleep(min(2.0 * (attempt + 1), 6.0))
                continue

    debug_path = write_llm_debug(
        stage_name=f"{task.name}_extraction",
        record_id=record_id,
        report_name=report_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_output=raw_output,
        error_message=str(last_error),
    )
    errors: List[str] = [f"llm_extraction_failed: {last_error}"]
    print(f"LLM extraction error ({task.name}): {last_error}")
    print(f"LLM debug saved to: {debug_path}")
    return {
        "fields": _default_fields(task),
        "schema_errors": errors,
        "raw_output": raw_output,
        "llm_called": True,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "reasoning": "",
        "evidence_quotes": [],
        "debug_path": str(debug_path),
        "attempts": attempts_used,
    }
