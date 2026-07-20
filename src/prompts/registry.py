"""
Prompt registry.

Loads prompt templates from ``prompts/<name>.txt`` and builds a schema-aware
system prompt for a task. Keeping prompts as files (not code) lets new tasks ship
their own instructions without touching the framework.
"""

from __future__ import annotations

from functools import lru_cache

from configs.config import PROMPTS_DIR
from configs.tasks.base import ExtractionTask


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """Read ``prompts/<name>.txt``. Raises ``FileNotFoundError`` if missing."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template missing: {path}")
    return path.read_text(encoding="utf-8")


def render_schema_block(task: ExtractionTask) -> str:
    """Human/LLM-readable description of the task output schema."""
    lines = ["Return ONLY a JSON object with exactly these fields:"]
    for f in task.fields:
        parts = [f'  "{f.name}": <{f.type}>']
        meta = []
        if f.required:
            meta.append("required")
        if f.type == "enum":
            meta.append("one of: " + ", ".join(f.enum))
        if f.default is not None:
            meta.append(f"default: {f.default}")
        if f.description:
            meta.append(f.description)
        if meta:
            parts.append("  # " + "; ".join(meta))
        lines.append("".join(parts))
    lines.append("Do not add extra keys or commentary. Output must be valid JSON.")
    return "\n".join(lines)


def build_system_prompt(task: ExtractionTask) -> str:
    """Combine the task's prompt template with the auto-generated schema block."""
    template = load_prompt(task.prompt_name) if task.prompt_name else ""
    schema_block = render_schema_block(task)
    if template:
        return f"{template.rstrip()}\n\n{schema_block}\n"
    return schema_block + "\n"
