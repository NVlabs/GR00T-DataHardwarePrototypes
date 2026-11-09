# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where recordings live on disk: one output folder.

Everything the data-collection workflow produces stays under one root::

    <root>/raw/{task}/{session}/episode_N/          recorded takes that passed the data check
    <root>/quarantine/{task}/{session}/episode_N/   takes that failed it
    <root>/aligned/{task}/{session}/episode_N/      time-aligned episodes
    <root>/reports/{task}/{session}/                data-check and health reports

The root is ``$UMI_OUTPUT_ROOT`` when set, else ``output/`` at the repository
root (git-ignored). A session recorded outside that layout (an explicit
``--data-dir`` elsewhere) uses sibling naming instead:
``<session>_quarantine``, ``<session>_aligned``, ``<session>_reports``.
"""

import os
from datetime import date
from pathlib import Path
from typing import Optional

OUTPUT_ROOT_ENV = "UMI_OUTPUT_ROOT"
KINDS = ("raw", "quarantine", "aligned", "reports")


def repo_root() -> Path:
    """The repository holding this file (the folder with pyproject.toml), else the cwd."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def output_root() -> Path:
    env = os.environ.get(OUTPUT_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "output"


def session_dir(kind: str, task_name: str, session_name: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown output kind {kind!r}; expected one of {KINDS}")
    task = (task_name or "").strip() or "untitled_task"
    return str(output_root() / kind / task / session_name)


def default_session_name(task_name: str, day: Optional[date] = None) -> str:
    task = (task_name or "").strip() or "untitled_task"
    return f"{(day or date.today()).isoformat()}_{task}"


def counterpart(session_path: str, kind: str) -> str:
    """The ``kind`` folder that belongs to the session at ``session_path``.

    ``session_path`` may be any of a session's folders. Inside the layout,
    ``<root>/{any kind}/{task}/{session}`` maps to ``<root>/{kind}/{task}/{session}``.
    Any other path maps through the sibling naming: ``<session>`` and
    ``<session>_{other kind}`` both map to ``<session>_{kind}`` (``<session>``
    itself for ``raw``).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown output kind {kind!r}; expected one of {KINDS}")
    path = Path(os.path.expanduser(session_path.rstrip(os.sep))).resolve()
    if path.parent.parent.name in KINDS:
        return str(path.parent.parent.parent / kind / path.parent.name / path.name)
    base = str(path)
    for other in KINDS[1:]:
        if base.endswith(f"_{other}"):
            base = base[: -len(other) - 1]
            break
    return base if kind == "raw" else f"{base}_{kind}"
