"""
Prompt registry.

Loads prompt templates from ``prompts/<name>.txt`` and builds a schema-aware
system prompt for a task. Schema-block instructions follow ``task.language``
(German for clinical tasks).
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
    if (task.language or "en").lower().startswith("de"):
        lines = [
            "Gib AUSSCHLIESSLICH ein JSON-Objekt mit genau diesen Feldern zurück:",
        ]
        footer = (
            "Keine zusätzlichen Keys und keinen Kommentartext. "
            "Die Ausgabe muss gültiges JSON sein. "
            "Alle Freitextwerte auf Deutsch."
        )
        required_label = "pflicht"
        one_of_label = "eines von"
        default_label = "default"
    else:
        lines = ["Return ONLY a JSON object with exactly these fields:"]
        footer = "Do not add extra keys or commentary. Output must be valid JSON."
        required_label = "required"
        one_of_label = "one of"
        default_label = "default"

    for f in task.fields:
        parts = [f'  "{f.name}": <{f.type}>']
        meta = []
        if f.required:
            meta.append(required_label)
        if f.type == "enum":
            meta.append(f"{one_of_label}: " + ", ".join(f.enum))
        if f.default is not None:
            meta.append(f"{default_label}: {f.default}")
        if f.description:
            meta.append(f.description)
        if meta:
            parts.append("  # " + "; ".join(meta))
        lines.append("".join(parts))
    lines.append(footer)
    return "\n".join(lines)


def build_system_prompt(task: ExtractionTask) -> str:
    """Combine the task's prompt template with the auto-generated schema block."""
    template = load_prompt(task.prompt_name) if task.prompt_name else ""
    schema_block = render_schema_block(task)
    if template:
        return f"{template.rstrip()}\n\n{schema_block}\n"
    return schema_block + "\n"
