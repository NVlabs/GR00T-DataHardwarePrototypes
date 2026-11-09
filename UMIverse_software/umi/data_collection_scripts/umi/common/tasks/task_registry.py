# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task registry — public API over the local task store.

Each TaskSpec captures everything that is fixed per-task (not per-session):
  - short_description:    one-sentence label written into recording metadata
    (zarr.json ``task_description``). Operator-facing; kept stable so downstream
    consumers don't break.
  - display_description:  free-text label for telling tasks apart in task
    lists. Independent of ``short_description``, so changing it never alters
    metadata written into recordings. Empty string is allowed if no extra hint
    is needed.
  - training_group:       canonical training-compatibility key. Two tasks sharing
    this value may be combined for training; differing values must not.
  - mode:                 "bimanual" | "unimanual"
  - HW_version:           hardware revision label (e.g. "v1", "v2", "v2.1")
  - use_head_mount:       default for the head-mount option
  - data_check_overrides: deep-merged on top of the global data_check_config.yaml
    when tooling resolves per-task checks. The training stage lives here too, as
    ``training_stage: {name: pretrain|midtrain|posttrain, version: vN}``; it is
    recorded into the session's ``meta`` and does not change any check. Absent
    means pretrain/v1.

The registry data itself lives in a local JSON file (``config/tasks.json``, else
the bundled ``tasks_fallback.json``; see ``task_registry_store``); this module
provides a lazy view over it. The Station Console adds a new task to that file
with just its name and ``short_description``; edit the file by hand to change a
task or fill in the optional fields. Never add session-specific values
(operator, device serials, date).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from umi.common.tasks import task_registry_store

DEFAULT_STAGE = "pretrain"
DEFAULT_STAGE_VERSION = "v1"


def stage_of(overrides: Optional[dict]) -> tuple[str, str]:
    """``(name, version)`` of ``overrides["training_stage"]``; pretrain/v1 when absent.

    Tolerant on the read side, like ``required_hours``: a malformed block
    reads as the default rather than breaking every ``get_task`` consumer.
    """
    block = (overrides or {}).get("training_stage")
    if not isinstance(block, dict):
        return DEFAULT_STAGE, DEFAULT_STAGE_VERSION
    name = block.get("name") or DEFAULT_STAGE
    version = block.get("version") or DEFAULT_STAGE_VERSION
    return str(name), str(version)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    short_description: str
    mode: str  # "bimanual" | "unimanual"
    # Require an explicit hardware revision so meta.HW_version is recorded for every
    # task.
    HW_version: str
    # Training-compatibility key. Same value across tasks => may be merged for
    # training; different values => keep separate. See module docstring.
    training_group: str
    # Optional dashboard label independent of recorded short_description.
    display_description: str = ""
    use_head_mount: bool = False
    has_trigger_cameras: bool = False
    data_check_overrides: dict = field(default_factory=dict)
    # Free-text instructions shown to the operator in the recording preview's
    # spare grid cell (\n for line breaks). Empty string = no instruction panel.
    operator_instruction: str = ""
    # Structured object entries provide name, count and optional contained
    # objects for the placement grid. An empty list falls back to parsing
    # operator_instruction text.
    objects: list = field(default_factory=list)
    # Target collection time is measured in hours. None means unset and must
    # remain distinct from a zero-hour target.
    required_hours: Optional[float] = None

    def training_stage(self) -> tuple[str, str]:
        """``(name, version)`` from ``data_check_overrides['training_stage']``."""
        return stage_of(self.data_check_overrides)


def _spec_from_entry(name: str, entry: dict) -> TaskSpec:
    """Materialize a ``TaskSpec`` from a raw JSON entry, tolerating extra keys."""
    fields = {
        "name": entry.get("name", name),
        "short_description": entry.get("short_description", ""),
        "mode": entry.get("mode", "bimanual"),
        "HW_version": entry.get("HW_version", ""),
        "training_group": entry.get("training_group", name),
        "display_description": entry.get("display_description", ""),
        "use_head_mount": bool(entry.get("use_head_mount", False)),
        "has_trigger_cameras": bool(entry.get("has_trigger_cameras", False)),
        "data_check_overrides": entry.get("data_check_overrides") or {},
        "operator_instruction": entry.get("operator_instruction", ""),
        "objects": entry.get("objects") or [],
    }
    required_hours = entry.get("required_hours")
    if required_hours is not None:
        # Treat malformed hand-edited targets as unset so registry consumers
        # remain usable.
        try:
            fields["required_hours"] = float(required_hours)
        except (TypeError, ValueError):
            pass
    return TaskSpec(**fields)


class _LazyTaskRegistry:
    """Dict-like proxy over the local task registry.

    Behaves like ``dict[str, TaskSpec]`` but lazily materializes ``TaskSpec``
    instances on first access. Always reads from the same registry singleton
    in ``task_registry_store``, so a successful ``reload()`` is visible here
    without restarting the process.
    """

    def _entries(self) -> dict:
        return task_registry_store.get_all_tasks()

    def __getitem__(self, key: str) -> TaskSpec:
        entries = self._entries()
        return _spec_from_entry(key, entries[key])

    def __contains__(self, key) -> bool:
        return key in self._entries()

    def __iter__(self):
        return iter(self._entries())

    def __len__(self) -> int:
        return len(self._entries())

    def keys(self):
        return self._entries().keys()

    def values(self):
        return [_spec_from_entry(n, e) for n, e in self._entries().items()]

    def items(self):
        return [(n, _spec_from_entry(n, e)) for n, e in self._entries().items()]

    def get(self, key: str, default=None):
        entries = self._entries()
        entry = entries.get(key)
        if entry is None:
            return default
        return _spec_from_entry(key, entry)

    def __repr__(self) -> str:
        return f"_LazyTaskRegistry(tasks={len(self)})"

    def __bool__(self) -> bool:
        return bool(self._entries())


TASK_REGISTRY: _LazyTaskRegistry = _LazyTaskRegistry()


def task_names() -> list[str]:
    """Sorted list of all registered task names."""
    return sorted(TASK_REGISTRY.keys())


def get_task(name: str) -> Optional[TaskSpec]:
    """Look up a task spec by name. Returns None if not found."""
    return TASK_REGISTRY.get(name)
