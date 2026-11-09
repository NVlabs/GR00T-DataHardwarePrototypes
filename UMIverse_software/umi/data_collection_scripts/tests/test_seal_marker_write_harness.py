# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The root harness's local seal-marker guard, verified end to end.

`session_sealed.json` is what makes `session_slot.slot_is_usable` refuse a
folder, so a marker written into a live session directory does not corrupt a
record -- it stops an operator's recorder, with no obvious cause on the
console. An unverified guard is not a guard, so this file checks it.

Deliberately at the `tests/` root: this is the floor every other test
directory actually gets, tested with no package guard underneath it.
`write_seal` has callers in several test directories.

Why the dangerous test is safe even with the guard mutated off: the refused
path points at a session directory that does not exist, and `write_seal`
returns None for a folder that is not there before it opens anything. So a
mutated-off guard turns the assertion from "raised" into "returned None" --
detected, with nothing written either way. A precondition is still checked,
because "the file it names cannot be created" is a property of today's
`write_seal` and not a promise it made.
"""

import pathlib

import pytest

from tests.conftest import SEAL_MARKER_GUARD_MARKER

MODULE_PATHS = [
    "umi.common.recording.session_seal",
    "station_ui.seal",
]


def _module(path):
    import importlib

    try:
        return importlib.import_module(path)
    except Exception:
        pytest.skip(f"{path} is not importable on this machine")


def _require_the_guard_before_sealing(module):
    """Refuse to attempt a seal unless the wrapper is demonstrably installed.

    A test that verifies a safety mechanism must not depend on that mechanism
    to stay safe. Keyed on the marker rather than the closure's `__name__`, so
    renaming the wrapper fails this loudly instead of quietly disarming it.
    """
    installed = getattr(module.write_seal, "guard_marker", None)
    if installed != SEAL_MARKER_GUARD_MARKER:
        pytest.fail(
            f"the root seal-marker harness is NOT active for {module.__name__} "
            "-- refusing to attempt a seal. This is the guard working: the "
            "test declines to perform the dangerous action rather than "
            "performing it unprotected."
        )


@pytest.mark.parametrize("path", MODULE_PATHS)
def test_every_module_that_holds_write_seal_is_wrapped(path):
    """`station_ui.seal` imports it by name, so the defining module is not
    enough -- that binding is the only one production actually calls."""
    _require_the_guard_before_sealing(_module(path))


def test_a_seal_under_the_temp_roots_is_written_rather_than_refused(tmp_path):
    """Accept-or-refuse, not refuse-everything.

    Every existing seal test writes a real marker under `tmp_path` and then
    reads it back. A guard that refused unconditionally would turn all of them
    into assertions about the harness, and leave `write_seal` itself never once
    exercised.
    """
    session_seal = _module("umi.common.recording.session_seal")
    _require_the_guard_before_sealing(session_seal)
    folder = tmp_path / "2026-08-27_umiV3_0001R_vive_task_a1b2c3d4e5f6"
    folder.mkdir()

    payload = session_seal.write_seal(folder, reason="operator", episode_count=1)

    assert payload is not None
    assert (folder / session_seal.SEAL_FILENAME).is_file()


@pytest.mark.parametrize("path", MODULE_PATHS)
def test_a_seal_aimed_at_the_real_local_data_cannot_land(path):
    """The end-to-end claim, down the real `write_seal` in each binding.

    The target is a session id nothing has ever recorded, under a real
    directory in the user's home (`~/Dev/UMI/local_data`) rather than a temp
    root. See this file's header for why the directory not existing is what
    makes running this safe.
    """
    module = _module(path)
    _require_the_guard_before_sealing(module)
    target = (
        pathlib.Path.home()
        / "Dev/UMI/local_data"
        / "2026-08-27_umiV3_0001R_vive_a_session_that_has_never_existed_ffffffffffff"
    )

    with pytest.raises(AssertionError) as excinfo:
        module.write_seal(target, reason="operator", episode_count=1)

    # Says what it is and what to do, so somebody who trips it does not have to
    # come and read this file first.
    assert "stops a recorder" in str(excinfo.value)
    assert "tmp_path" in str(excinfo.value)
    assert not (target / "session_sealed.json").exists()


def test_the_tilde_form_is_expanded_before_it_is_judged():
    """`seal_path` expands a leading `~`, so the guard has to judge the expanded
    path: one that compared the raw text would let the tilde form of a real
    session directory through."""
    session_seal = _module("umi.common.recording.session_seal")
    _require_the_guard_before_sealing(session_seal)

    with pytest.raises(AssertionError):
        session_seal.write_seal(
            "~/Dev/UMI/local_data/2026-08-27_never_existed_ffffffffffff",
            reason="operator",
            episode_count=1,
        )
