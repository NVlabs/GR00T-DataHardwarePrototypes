# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the active data-check config from an immutable baseline.

`config/data_check/data_check_config.yaml` is version-controlled, so handing
the recorder that path means every setting an operator changes dirties the
station's checkout, conflicts on the next `git pull`, and overwrites the
shipped defaults with no way left to tell "what UMI ships" from "what this
bench chose".

The recording config already solved this: `station_ui.core.config_assembler`
only ever *reads* its template and writes the merged result under the station
state dir. This is the same move for data-check -- the baseline stays read-only
in the checkout, and the generated per-task file lands beside the generated
recording config rather than beside its own baseline.

The registry's own `data_check_overrides` are folded in here rather than left
for `RecorderManager._save_data_check_config` to apply, so the generated file
states what will actually be enforced. That manager applies the same merge to
whatever it is given, and merging an already-merged config is a no-op, so the
two agree instead of the artifact under-reporting the real thresholds.
"""

from __future__ import annotations

import os
import pathlib
import re
import tempfile

import yaml

from station_ui import fast_yaml, settings
from station_ui.errors import StationUIError

# Use RecorderManager's config merge so generated checks match the
# thresholds the recorder applies.
from umi.common.recording.session_slot import _deep_merge_dict

# Write generated check configs under the state directory; keep the tracked
# baseline repository-relative and read-only.
GENERATED_DIR = ("data_check",)
BASELINE_DIR = ("config", "data_check")
BASELINE_NAME = "data_check_config.yaml"

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


class BaselineMissing(StationUIError):
    """The read-only baseline this config would be generated from is not there.

    A `StationUIError` so a route answers it beside the other expected
    failures. Left as a bare `FileNotFoundError` it surfaces as a 500 naming
    an internal path, when what the operator needs to be told is that the
    checkout is incomplete.
    """


def _slug(value, fallback: str) -> str:
    """One path component, with nothing in it that can address another one.

    A task name arrives from a free text field, and both halves of the
    generated path are interpolated into a filesystem location. Reduced to its
    final component first so `../../etc/passwd` cannot climb out before the
    character filter ever sees it.
    """
    name = pathlib.PurePosixPath(str(value or "").strip().replace("\\", "/")).name
    return _UNSAFE.sub("", name) or fallback


def baseline_path(repo_root) -> pathlib.Path:
    return pathlib.Path(repo_root).joinpath(*BASELINE_DIR, BASELINE_NAME)


def output_path(state_dir, task_name) -> pathlib.Path:
    """Where this task's generated config lives.

    Rooted at the state dir, not the checkout. `state_dir=None` resolves the
    default from the environment, which is what the recorder handoff in
    `routes/record.py` wants: an absolute path that does not depend on the
    working directory the child happens to be launched in.
    """
    root = (
        settings.state_dir_from_environment()
        if state_dir is None
        else pathlib.Path(state_dir)
    )
    return root.joinpath(*GENERATED_DIR, f"{_slug(task_name, 'unnamed')}.yaml")


def _task_spec(name: str):
    """The registry's entry for *name*, or None if it has nothing to say.

    A module-level function, not an inline import, so a test can replace it:
    the registry is read from a local file, and no test may reach for it.
    Every failure -- unregistered task, unreadable file -- is the same answer
    here, because all of them mean "generate from the baseline
    alone" and none of them should stop an operator recording.
    """
    try:
        from umi.common.tasks.task_registry import get_task

        return get_task(name)
    except Exception:
        return None


def generate(repo_root, *, task_name, state_dir=None) -> pathlib.Path:
    """Write this task's data-check config, and return where it landed.

    Two roots on purpose: the baseline is read from the checkout, the result is
    written under the state dir.
    """
    baseline = baseline_path(repo_root)
    try:
        config = fast_yaml.safe_load(baseline.read_text()) or {}
    except FileNotFoundError as exc:
        raise BaselineMissing(
            f"no data-check baseline at {baseline} — "
            f"{baseline.name} is version-controlled and should be in the checkout"
        ) from exc

    spec = _task_spec(task_name)
    if spec is not None:
        # default -> task, the same order the session snapshot uses.
        overrides = getattr(spec, "data_check_overrides", None) or {}
        if overrides:
            config = _deep_merge_dict(config, overrides)

    out = output_path(state_dir, task_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out, yaml.dump(config, default_flow_style=False, sort_keys=False))
    return out


def _atomic_write(out: pathlib.Path, text: str) -> None:
    """Replace the file whole, never leave a partial one in its place.

    The path is handed to the recorder as a `click.Path(exists=True)`, which a
    half-written file satisfies -- and then fails to parse deep inside the
    recorder, after the operator has already started the session.
    """
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
