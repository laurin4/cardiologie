"""
Task registry.

New extraction tasks are added by creating ``configs/tasks/<name>/task.py`` that
exposes a module-level ``TASK`` of type :class:`ExtractionTask`, then registering
the name below (or relying on the directory-based auto-discovery).
"""

from __future__ import annotations

import importlib
from typing import Dict, List

from configs.tasks.base import (  # re-export for convenience
    EvidenceGroup,
    ExtractionTask,
    SchemaField,
)

__all__ = [
    "EvidenceGroup",
    "ExtractionTask",
    "SchemaField",
    "load_task",
    "available_tasks",
]

# Explicit registry of task package names shipped with the framework.
_REGISTERED_TASKS: List[str] = ["demo_extraction"]

_CACHE: Dict[str, ExtractionTask] = {}


def available_tasks() -> List[str]:
    """Return the list of registered task names."""
    return list(_REGISTERED_TASKS)


def load_task(name: str) -> ExtractionTask:
    """
    Load a registered task by name.

    Looks up ``configs.tasks.<name>.task`` and returns its module-level ``TASK``.
    """
    if name in _CACHE:
        return _CACHE[name]
    if name not in _REGISTERED_TASKS:
        raise KeyError(
            f"Unknown extraction task '{name}'. Registered tasks: {_REGISTERED_TASKS}"
        )
    module = importlib.import_module(f"configs.tasks.{name}.task")
    task = getattr(module, "TASK", None)
    if not isinstance(task, ExtractionTask):
        raise TypeError(
            f"Task module 'configs.tasks.{name}.task' must define a module-level "
            f"'TASK' of type ExtractionTask."
        )
    _CACHE[name] = task
    return task
