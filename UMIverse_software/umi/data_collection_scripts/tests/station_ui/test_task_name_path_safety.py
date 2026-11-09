# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The free-text Task name is a filesystem path component.

`build_data_dir` places the operator's task name into the session folder path,
`<output>/raw/<task>/<session>`, which becomes `recording.data_dir`. The Task
field is a plain `<input type="text">` and is never checked against the
registry (unregistered tasks are supported on purpose), so a name may contain a
`/` or a `..`. Placed in the path unchanged, it would split the folder (or,
with `..`, escape the output root entirely), and on reopen the recording date
would fall out of the first ten characters of the name, silently forking the
session across two days.

The task therefore goes through `sanitize_task_name` before it enters the path.
These tests pin that it cannot add a path segment or climb out of the root,
while a normal name is still slugged and kept and a blank one still adds no
task segment to the session name (rather than the `unnamed` slug).
"""

from pathlib import Path

from station_ui.core.config_utils import build_data_dir
from umi.common.utility.output_layout import output_root

# A valid 12-char lowercase-hex session uid: `build_data_dir` refuses anything
# else, and the suffix is not what these tests are about.
UID = "0123456789ab"
DATE = "2026-08-06"


def _below_raw(path: str) -> tuple[str, ...]:
    """The path's components under `<output>/raw`; fails if it is not there."""
    return Path(path).relative_to(output_root() / "raw").parts


def test_a_slash_in_the_task_name_adds_no_path_segment():
    path = build_data_dir(DATE, "pick/place", "bimanual", session_uid=UID)
    # One task folder and one session folder; a task that split the folder
    # would add a third component.
    parts = _below_raw(path)
    assert len(parts) == 2
    assert parts[1].startswith(DATE), "the date must stay date-first"


def test_a_dotdot_task_name_cannot_climb_out_of_the_data_root():
    path = build_data_dir(DATE, "../../tmp/evil", "bimanual", session_uid=UID)
    parts = _below_raw(path)
    assert len(parts) == 2, "a `..` task escaped the data root"
    assert ".." not in path
    assert parts[1].startswith(DATE)


def test_a_normal_task_name_is_slugged_and_kept():
    path = build_data_dir(DATE, "Plane Assembly v9", "bimanual", session_uid=UID)
    assert "Plane_Assembly_v9" in path
    assert path.endswith(UID)


def test_a_blank_task_name_leaves_no_task_segment():
    # `sanitize_task_name("")` returns "unnamed"; the folder must not gain that
    # word -- a blank task drops its segment.
    path = build_data_dir(DATE, "   ", "bimanual", session_uid=UID)
    assert "unnamed" not in path
    assert path.rsplit("/", 1)[-1].startswith(DATE)
