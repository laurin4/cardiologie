"""Robust parsing of JSON objects out of raw LLM text output."""

import json
import os
import re
from typing import Any, Dict


def _debug_llm_output_enabled() -> bool:
    """True when DEBUG_LLM_OUTPUT is set to 1/true/yes (default: off)."""
    return os.environ.get("DEBUG_LLM_OUTPUT", "").strip().lower() in ("1", "true", "yes")


def parse_llm_json_output(raw_output: str, context_name: str) -> Dict[str, Any]:
    """
    Parse the first JSON object from an LLM response.

    Handles markdown code fences and trailing prose. Raises ``ValueError`` with the
    ``context_name`` when no valid JSON object can be recovered.
    """
    if _debug_llm_output_enabled():
        print(f"=== RAW LLM OUTPUT ({context_name}) START ===")
        print(raw_output)
        print(f"=== RAW LLM OUTPUT ({context_name}) END ===")

    if not raw_output or not raw_output.strip():
        raise ValueError(f"Empty LLM response in {context_name}: no JSON content.")

    text = raw_output.strip()

    # Strip markdown code fences.
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 1) Try direct parse.
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Extract the first JSON object substring.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception as exc:
            raise ValueError(
                f"Could not robustly parse JSON in [{context_name}]: {exc}"
            )

    raise ValueError(f"No JSON structure found in the LLM response [{context_name}].")
