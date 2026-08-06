"""
Task specification for the clinical extraction framework.

An extraction *task* is the pluggable unit that adapts the generic pipeline to a
concrete extraction problem. It bundles everything the stages need:

- ``fields``            -> schema for LLM output + validation (types, enums, required)
- ``evidence_groups``   -> keyword groups for rule-based evidence extraction
- ``section_markers``   -> optional section headings recognised in report text
- ``negation_patterns`` -> optional sentence-local negation regexes
- ``prompt_name``       -> prompt template file under ``prompts/``
- ``consistency_rules`` -> declarative guardrail / consistency checks

This module has no dependency on ``src`` so it can be imported by every stage
without creating import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

# Roles an evidence group can play in the rule layer.
ROLE_POSITIVE = "positive"   # supports extraction; sent to the LLM
ROLE_CONTEXT = "context"     # contextual hint; sent to the LLM
ROLE_NEGATION = "negation"   # explicit negation/exclusion; not "positive" on its own
VALID_ROLES = (ROLE_POSITIVE, ROLE_CONTEXT, ROLE_NEGATION)

# Supported schema field types.
FIELD_TYPES = ("string", "boolean", "integer", "number", "enum", "array")


@dataclass(frozen=True)
class SchemaField:
    """One field in a task output schema."""

    name: str
    type: str = "string"
    required: bool = False
    enum: Tuple[str, ...] = ()
    default: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.type not in FIELD_TYPES:
            raise ValueError(
                f"SchemaField '{self.name}': unsupported type '{self.type}'. "
                f"Allowed: {FIELD_TYPES}"
            )
        if self.type == "enum" and not self.enum:
            raise ValueError(f"SchemaField '{self.name}': enum type requires 'enum' values.")


@dataclass(frozen=True)
class EvidenceGroup:
    """A named keyword group used by the rule-based evidence extractor."""

    name: str
    phrases: Tuple[str, ...]
    role: str = ROLE_POSITIVE
    priority: int = 100

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"EvidenceGroup '{self.name}': invalid role '{self.role}'. Allowed: {VALID_ROLES}"
            )


@dataclass(frozen=True)
class VariableSpec:
    """
    One clinical variable extracted with its own LLM call.

    When a parent :class:`ExtractionTask` declares ``variables``, the pipeline
    runs one prompt/schema/keyword subset per variable and merges the results
    into a single patient row.
    """

    name: str
    label: str
    prompt_name: str
    fields: Tuple[SchemaField, ...]
    evidence_group_names: Tuple[str, ...]
    consistency_rules: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ExtractionTask:
    """A complete, pluggable extraction task specification."""

    name: str
    description: str = ""
    fields: Tuple[SchemaField, ...] = ()
    evidence_groups: Tuple[EvidenceGroup, ...] = ()
    section_markers: Tuple[Tuple[str, str], ...] = ()
    negation_patterns: Tuple[Tuple[str, str], ...] = ()
    prompt_name: str = ""
    consistency_rules: Tuple[Dict[str, Any], ...] = ()
    language: str = "en"
    # When True and rule evidence finds nothing actionable, still call the LLM
    # with the cleaned full report text (useful for smoke tests / sparse keywords).
    send_full_text_when_no_evidence: bool = False
    # Optional: one LLM call per clinical variable (merged into one output row).
    variables: Tuple[VariableSpec, ...] = ()

    def field_by_name(self, name: str) -> SchemaField | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def schema_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view of the output schema (for prompts / docs)."""
        return {
            "task": self.name,
            "fields": [
                {
                    "name": f.name,
                    "type": f.type,
                    "required": f.required,
                    "enum": list(f.enum),
                    "default": f.default,
                    "description": f.description,
                }
                for f in self.fields
            ],
        }
