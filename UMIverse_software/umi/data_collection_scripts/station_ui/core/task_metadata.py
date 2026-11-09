# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the task registry contributes to a session, in one place.

Two different files need a value from ``umi.common.tasks.task_registry`` at the
moment a config is saved, so :func:`task_metadata` lives here and every config
writer calls it rather than copying it.

:func:`task_metadata` returns ``short_description`` and ``HW_version``, which
become ``meta.task_description`` / ``meta.HW_version`` and from there every
episode's zarr attrs and every LeRobot ``annotation.human.task_description``.
An unknown task yields ``""`` and never the template's placeholder
``"Description"``: the placeholder is *truthy*, so
``real_scripts/data_collection/record_data.py`` skips its ``input()`` fallback
and labels the whole session "Description". Blank is falsy, so the recorder
correctly falls back to asking.

Best-effort about the *environment* (a registry lookup that fails) and exact
about the *data*: a task that is registered gets its real values or nothing at
all, never a plausible default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from umi.common.tasks import task_registry as _registry

logger = logging.getLogger(__name__)


def _stage_of(overrides):
    return _registry.stage_of(overrides)


__all__ = [
    "TaskMetadata",
    "task_metadata",
]


@dataclass(frozen=True)
class TaskMetadata:
    """Registry values that belong in ``meta``. Blank when the task is unknown."""

    description: str = ""
    hw_version: str = ""
    # The task's training stage, recorded into ``meta``; blank when unknown.
    training_stage: str = ""
    training_stage_version: str = ""


def _spec(task_name):
    if not task_name:
        return None
    try:
        return _registry.get_task(task_name)
    except Exception:  # pragma: no cover - a registry outage is not a save error
        logger.debug("task registry lookup failed for %r", task_name, exc_info=True)
        return None


def task_metadata(task_name) -> TaskMetadata:
    """``short_description`` / ``HW_version`` for *task_name*, or blanks."""
    spec = _spec(task_name)
    if spec is None:
        return TaskMetadata()
    stage = _stage_of(getattr(spec, "data_check_overrides", None))
    return TaskMetadata(
        description=spec.short_description or "",
        hw_version=spec.HW_version or "",
        training_stage=stage[0],
        training_stage_version=stage[1],
    )
