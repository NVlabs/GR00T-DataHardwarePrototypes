# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the recorder should know, say and record about a task-registry lookup.

Kept apart from ``record_data.py`` so the rules live next to the registry they
describe and can be tested without the recorder's hardware setup. Nothing here
prints or draws; :func:`resolve_task_spec` takes a ``log`` callback for the
one thing worth telling the station log, and :func:`panel_with_warning`
returns a panel spec for the visualizer to render.

Two registry gaps leave episodes with no setup text and would otherwise be
silent at record time: the task is not in the registry at all, or the
registry came from an unexpected source. :func:`registry_warning` names
whichever applies, and :func:`registry_provenance` stamps the facts onto every
saved episode so a missing instruction can be explained afterwards instead of
guessed at. A task with no operator instruction is not a gap: setup text is
optional, and the Station Console adds tasks without it.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from umi.common.tasks import task_registry, task_registry_store

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_task_spec",
    "registry_warning",
    "panel_with_warning",
    "registry_provenance",
]

WARNING_HEADER_COLOR = (80, 80, 255)


def resolve_task_spec(task_name, log: Optional[Callable[[str], None]] = None):
    """The task's registry entry, with where the registry came from.

    Returns ``(spec, info, retried)``. ``spec`` is ``None`` for a task the
    registry does not carry; ``info`` is ``task_registry_store.get_registry_info``
    (``source`` is ``file:<path>`` or ``bundled``); ``retried`` says the
    lookup missed once and the registry file was re-read before the second
    look.

    The registry is loaded once per process; re-reading it on a miss picks up
    a task added to the file after the process started.

    ``log`` receives one line when the re-read or the info lookup fails.
    Defaults to this module's logger.
    """
    say = log or logger.warning
    spec = task_registry.get_task(task_name)
    retried = False
    if spec is None:
        retried = True
        try:
            task_registry_store.reload()
            spec = task_registry.get_task(task_name)
        except Exception as exc:  # noqa: BLE001 - the banner reports it; the log says why
            say(f"[Registry] re-read for {task_name!r} failed: {exc!r}")
    try:
        info = dict(task_registry_store.get_registry_info())
    except Exception as exc:  # noqa: BLE001
        say(f"[Registry] no registry info: {exc!r}")
        info = {}
    return spec, info, retried


def registry_warning(task_name, spec, info) -> str | None:
    """What to warn about the registry lookup, or ``None`` when all is well.

    Worst first: task missing, then a stale copy.
    """
    source = task_registry_store.source_label(info.get("source"))
    stamp = f"task list from {source or 'unknown'}, updated {info.get('updated_at', '?')}, v{info.get('version', '?')}"
    if spec is None:
        return f"TASK NOT IN REGISTRY: {task_name}\nNo setup text will be recorded.\n({stamp})"
    if source is not None and task_registry_store.is_stale_copy(source):
        return f"REGISTRY CAME FROM AN UNEXPECTED SOURCE: {source}.\nSetup text may be out of date.\n({stamp})"
    return None


def panel_with_warning(panel: dict | None, warning: str | None) -> dict:
    """The operator-instruction panel spec to show.

    ``panel`` is the instruction panel (or ``None`` when the task has no
    setup text); ``warning`` is :func:`registry_warning`'s text (or ``None``).
    A warning goes above the instruction, never instead of it: a cached copy
    may still carry text for the task, and hiding it would trade one gap for
    another. With no instruction the warning is the whole panel. With neither
    the caller keeps its empty default.
    """
    if not warning:
        return dict(panel or {})
    if panel is None:
        return {
            "text": f"REGISTRY WARNING\n{warning}",
            "title": "Operator Instruction",
            "header_lines": ["REGISTRY WARNING"],
            "header_color": WARNING_HEADER_COLOR,
        }
    out = dict(panel)
    out["text"] = f"REGISTRY WARNING\n{warning}\n\n" + panel["text"]
    out["header_lines"] = ["REGISTRY WARNING"] + list(panel.get("header_lines") or [])
    return out


def registry_provenance(spec, info, retried) -> dict:
    """What the episode records about the registry it was recorded under."""
    return {
        "source": info.get("source"),
        "version": info.get("version"),
        "updated_at": info.get("updated_at"),
        "task_registered": spec is not None,
        "has_operator_instruction": bool(
            spec and (spec.operator_instruction or "").strip()
        ),
        "refetched": bool(retried),
    }
