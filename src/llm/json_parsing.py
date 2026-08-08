"""Robust parsing of JSON objects out of raw LLM text output."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


def _debug_llm_output_enabled() -> bool:
    """True when DEBUG_LLM_OUTPUT is set to 1/true/yes (default: off)."""
    return os.environ.get("DEBUG_LLM_OUTPUT", "").strip().lower() in ("1", "true", "yes")


_USZ_TOKENS = (
    "<start_of_turn>user",
    "<start_of_turn>model",
    "<start_of_turn>",
    "<end_of_turn>",
    "<think>",
    "</think>",
)


def _strip_noise(text: str) -> str:
    out = text.strip()
    for tok in _USZ_TOKENS:
        out = out.replace(tok, "")
    # Markdown fences (opening / closing).
    out = re.sub(r"^```(?:json)?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*```$", "", out)
    return out.strip()


def _extract_balanced_object(text: str) -> Optional[str]:
    """Return first top-level `{...}` using brace matching inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _loads_lenient(json_str: str) -> Dict[str, Any]:
    """Parse JSON; tolerate trailing commas in objects/arrays."""
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            return obj
        raise ValueError("JSON root is not an object")
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
        raise ValueError("JSON root is not an object")


def parse_llm_json_output(raw_output: str, context_name: str) -> Dict[str, Any]:
    """
    Parse the first JSON object from an LLM response.

    Handles markdown fences, USZ/Gemma turn markers, trailing commas, and prose
    around the object. Raises ``ValueError`` with *context_name* when no valid
    JSON object can be recovered.
    """
    if _debug_llm_output_enabled():
        print(f"=== RAW LLM OUTPUT ({context_name}) START ===")
        print(raw_output)
        print(f"=== RAW LLM OUTPUT ({context_name}) END ===")

    if not raw_output or not str(raw_output).strip():
        raise ValueError(f"Empty LLM response in {context_name}: no JSON content.")

    text = _strip_noise(str(raw_output))

    # 1) Direct parse.
    try:
        return _loads_lenient(text)
    except Exception:
        pass

    # 2) Balanced brace extraction (more reliable than greedy regex).
    candidate = _extract_balanced_object(text)
    if candidate:
        try:
            return _loads_lenient(candidate)
        except Exception as exc:
            raise ValueError(
                f"Could not robustly parse JSON in [{context_name}]: {exc}"
            ) from exc

    # 3) Last resort: greedy regex (legacy behaviour).
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return _loads_lenient(match.group(0))
        except Exception as exc:
            raise ValueError(
                f"Could not robustly parse JSON in [{context_name}]: {exc}"
            ) from exc

    raise ValueError(f"No JSON structure found in the LLM response [{context_name}].")
