# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local loader for the UMI task registry.

The registry lists every task a station can record (description, operator
instruction, objects, data-check overrides). It is read from a local file,
the same way as the device registry (``umi.utils.device_registry``):

1. this station's own file -- ``$UMI_TASK_REGISTRY_FILE`` when set, else
   ``config/tasks.json`` at the repository root;
2. ``umi/common/tasks/tasks_fallback.json`` -- the bundled list, empty so a new
   station starts with no tasks -- when that file does not exist or cannot be
   read.

The registry is loaded lazily on first access.

Nobody has to register a task by hand. The Station Console calls
:func:`add_task` when an operator applies a session under a name the list does
not carry, which appends it to this station's own file. The first such task
creates that file from the list the station already offers, so the bundled
tasks stay selectable beside it.

JSON layout::

    {
      "version": <int>,
      "updated_at": "<ISO8601 UTC>",
      "tasks": {
        "<task_name>": {
          "name": "...",
          "short_description": "...",
          "display_description": "...",
          "mode": "bimanual" | "unimanual",
          "HW_version": "...",
          "training_group": "...",
          "use_head_mount": <bool>,
          "has_trigger_cameras": <bool>,
          "data_check_overrides": { ..., "training_stage": {"name": "...", "version": "v1"} },
          "operator_instruction": "...",
          "objects": [{"name": "...", "count": 1, "contains": [ ... ]}, ...],
          "required_hours": null | <float>   # DC target in hours; null = no target
        },
        ...
      }
    }
"""

import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from umi.common.utility.output_layout import repo_root

logger = logging.getLogger(__name__)

ENV_REGISTRY_FILE = "UMI_TASK_REGISTRY_FILE"
FALLBACK_FILE = Path(__file__).parent / "tasks_fallback.json"

# A directly assigned _REGISTRY override remains active until reload().
_REGISTRY: Optional[dict] = None
_REGISTRY_SOURCE: Optional[str] = None


def local_registry_path() -> Path:
    """This station's own task registry: ``$UMI_TASK_REGISTRY_FILE``, else
    ``config/tasks.json``."""
    env = os.environ.get(ENV_REGISTRY_FILE, "").strip()
    if env:
        return Path(env).expanduser()
    return repo_root() / "config" / "tasks.json"


def _empty_registry() -> dict:
    return {"version": 0, "updated_at": "1970-01-01T00:00:00Z", "tasks": {}}


def _resolve_registry() -> tuple[dict, str]:
    """Run the load order and return ``(data, source)``.

    Source is ``file:<path>`` for this station's own file, and ``bundled`` for
    the snapshot shipped with the code.
    """
    path = local_registry_path()
    if path.is_file():
        try:
            return json.loads(path.read_text()), f"file:{path}"
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Task registry %s unreadable (%s); using the snapshot", path, e
            )
    try:
        return json.loads(FALLBACK_FILE.read_text()), "bundled"
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to read task registry snapshot %s: %s", FALLBACK_FILE, e)
        return _empty_registry(), "bundled"


def load_registry() -> dict:
    """Load the task registry from its local file."""
    global _REGISTRY_SOURCE
    data, _REGISTRY_SOURCE = _resolve_registry()
    return data


def _get_registry() -> dict:
    """The lazily-loaded registry payload."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def reload() -> dict:
    """Re-read the registry file, replacing the in-memory copy."""
    global _REGISTRY
    _REGISTRY = load_registry()
    return _REGISTRY


def add_task(name: str, short_description: str) -> Path:
    """Append a task to this station's own file and make it active.

    A name the file already carries is left as it is. Raises when the file
    exists but cannot be read: the loader falls back to the snapshot in that
    case, and writing that fallback out would replace the station's own list.
    """
    path = local_registry_path()
    if path.is_file():
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("tasks", {}), dict):
            raise ValueError(f"{path} does not hold a task list")
    else:
        # Initialize a new local task registry from the tasks already available.
        data = {
            "version": get_registry_info().get("version") or 1,
            "tasks": copy.deepcopy(get_all_tasks()),
        }
    tasks = data.setdefault("tasks", {})
    if name not in tasks:
        tasks[name] = {"name": name, "short_description": short_description}
        data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    reload()
    return path


def get_all_tasks() -> dict:
    """Return the raw ``tasks`` dict from the registry payload."""
    return _get_registry().get("tasks", {})


# Recognize local-file and bundled-snapshot task registries as supported sources.
_LIVE_SOURCE_PREFIXES = ("file", "bundled")


def source_label(source) -> Optional[str]:
    """Normalise a ``get_registry_info()["source"]`` value.

    The one place that decides what "no source" looks like: ``None``, ``""``,
    whitespace and the store's own ``"unknown"`` placeholder all come back as
    ``None``; anything else comes back stripped. Callers that show the label
    to a person choose their own wording for ``None`` (the recorder says
    "unknown" in its stamp, the station card says nothing) but they never
    have to agree on which raw values mean "missing".
    """
    text = str(source or "").strip()
    if not text or text == "unknown":
        return None
    return text


def is_stale_copy(source) -> bool:
    """True when *source* (``get_registry_info()["source"]``) is not one the
    store produces (a local file or the bundled snapshot).

    Deliberately a deny list: a label this function has never seen warns
    rather than passes, so an unexpected source errs toward a loud false
    alarm. ``None`` / ``""`` / ``"unknown"`` mean the source was not recorded
    at all, which is not evidence of staleness and does not warn.
    """
    label = source_label(source)
    if label is None:
        return False
    return not label.startswith(_LIVE_SOURCE_PREFIXES)


def get_registry_info() -> dict:
    """Return metadata about the registry (version, updated_at, source)."""
    reg = _get_registry()
    return {
        "version": reg.get("version", 0),
        "updated_at": reg.get("updated_at", "unknown"),
        "source": _REGISTRY_SOURCE or "unknown",
        "task_count": len(reg.get("tasks", {})),
    }
