# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repo-wide floor: no test, anywhere, can touch this station's real state.

Everything the data-collection workflow writes or reads lives locally:

  * the output folder (`umi.common.utility.output_layout`) -- `output/` at the
    repository root unless `UMI_OUTPUT_ROOT` moves it;
  * the device registry (`config/devices.json`, `UMI_DEVICE_REGISTRY_FILE`);
  * the task registry (`config/tasks.json`, `UMI_TASK_REGISTRY_FILE`).

`_isolated_local_state` points all three at temp locations for every test, so
no test writes a recording into the checkout's `output/` and no test reads this
station's own registries -- the suite sees the bundled snapshots instead, on
every machine alike.

The local seal marker. `session_sealed.json` is the one file in this system
that makes `session_slot.slot_is_usable` refuse a folder. A test that writes one
into a live session directory does not corrupt a record -- it stops a recorder,
and the operator's console simply declines the session with no obvious cause.
`write_seal` has callers in several test directories, so the guard is here.

Refuse-only: the destination is an argument the caller chose, so there is
nothing to redirect -- silently rewriting it would make every seal test assert
against a path its subject never used. So the wrapper allows the run's temp
roots and fails on anything else. Verified by `test_seal_marker_write_harness.py`,
which walks the real `write_seal` down both paths.
"""

import importlib
import os
import pathlib
import tempfile

import pytest


def _import(path):
    try:
        return importlib.import_module(path)
    except Exception:  # noqa: BLE001 - an optional module missing here is not a failure
        return None


def _reload_registries():
    for path in ("umi.utils.device_registry", "umi.common.tasks.task_registry_store"):
        module = _import(path)
        if module is not None and hasattr(module, "reload"):
            try:
                module.reload()
            except Exception:  # noqa: BLE001 - a test may have left a bad file on purpose
                pass


@pytest.fixture(autouse=True)
def _isolated_local_state(monkeypatch, tmp_path_factory):
    """Point the output folder and both registries at temp locations.

    The registry files are deliberately paths that do not exist, so the loaders
    fall back to the bundled snapshots; a test that wants its own registry
    writes the file and sets the variable itself. The output root gets a
    directory of its own rather than a subdirectory of `tmp_path`, so a test
    that enumerates `tmp_path` does not count it.
    """
    root = tmp_path_factory.mktemp("umi-local-state")
    monkeypatch.setenv("UMI_OUTPUT_ROOT", str(root / "output"))
    monkeypatch.setenv("UMI_DEVICE_REGISTRY_FILE", str(root / "no-devices.json"))
    monkeypatch.setenv("UMI_TASK_REGISTRY_FILE", str(root / "no-tasks.json"))
    _reload_registries()
    yield root
    monkeypatch.undo()
    _reload_registries()


# ---------------------------------------------------------------------------
# No test writes the local seal marker outside the run's temp roots.
# ---------------------------------------------------------------------------

# Patch every imported write_seal binding, including station_ui.seal, so
# no caller can seal a real recording folder during tests.
_SEAL_MARKER_MODULES = (
    "umi.common.recording.session_seal",  # defines it
    "station_ui.seal",  # imports it by name
)

# Stamped on the wrapper so a test can recognise it structurally rather than
# by the wrapper's name.
SEAL_MARKER_GUARD_MARKER = "umi-seal-marker-guard"


def _seal_marker_roots(basetemp):
    """Where a test is allowed to seal: the run's temp roots, and nothing else.

    Both, not just pytest's: `tmp_path` and `tmp_path_factory` land under
    `basetemp`, while a test using `tempfile.TemporaryDirectory` lands under
    the system temp dir, and `--basetemp` moves the first without moving the
    second. Realpath'd because a temp dir reached through a symlink would
    otherwise read as outside its own root.
    """
    roots = [basetemp, pathlib.Path(tempfile.gettempdir())]
    return tuple(pathlib.Path(os.path.realpath(root)) for root in roots)


@pytest.fixture(autouse=True)
def _no_test_seals_a_live_session_folder(monkeypatch, tmp_path_factory):
    """Refuse a `write_seal` aimed anywhere but the run's temp roots.

    The stakes are in this file's header: `session_sealed.json` is what makes
    `slot_is_usable` refuse a folder, so one written into a live session
    directory stops a recorder rather than merely dirtying a directory, and
    stray test folders there would be read by `count_episodes` as real
    sessions.

    Refuse-only rather than redirect-first, for the reason the header gives:
    the destination is the caller's argument, not a constant the code looks up,
    so there is nothing to point at scratch without lying to the test about
    where its subject wrote.
    """
    real = _import(_SEAL_MARKER_MODULES[0])
    if real is None or not hasattr(real, "write_seal"):
        return
    original = real.write_seal
    roots = _seal_marker_roots(tmp_path_factory.getbasetemp())

    def _guarded(folder, **kwargs):
        target = pathlib.Path(os.path.realpath(os.path.expanduser(str(folder))))
        if not any(target == root or root in target.parents for root in roots):
            raise AssertionError(
                f"a test tried to write a session seal marker into {target} -- "
                "outside this run's temp roots. session_sealed.json is the one "
                "file that makes session_slot.slot_is_usable REFUSE a folder, "
                "so one written into a live session directory stops a recorder. "
                "Seal under tmp_path (or a tmp_path-rooted $HOME, as "
                "tests/station_ui/conftest.py does). See tests/conftest.py."
            )
        return original(folder, **kwargs)

    _guarded.guard_marker = SEAL_MARKER_GUARD_MARKER

    for path in _SEAL_MARKER_MODULES:
        module = _import(path)
        if module is None or not hasattr(module, "write_seal"):
            continue
        monkeypatch.setattr(module, "write_seal", _guarded)


# ---------------------------------------------------------------------------
# A log handler must not outlive the directory it writes into.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_test_leaves_a_file_log_on_a_shared_logger():
    """Detach handlers a test attached to a module-level logger.

    `background_aligner._worker_main` opens a `logging.FileHandler` under its
    `log_path` directory and adds it to `umi.common.data.data_buffer` and
    `real_scripts.data_processing.align_data`, then never removes it. In
    production that is correct and costs nothing: the worker is a process, and
    the handler dies with it.

    Several tests call `_worker_main` in-process or in a thread, and there the
    handler outlives the test while its directory -- a `TemporaryDirectory` --
    does not. The logger is global, so the next test in that process to log
    through `data_buffer` reopens a path under a deleted directory and dies with
    `FileNotFoundError`, having touched none of this.

    Serial runs can hide it, because the directory order happens to run the
    affected tests first; splitting the files across xdist workers drops that
    ordering and exposes it in unrelated tests.

    Autouse and repo-wide rather than local to the aligner tests: a cleanup
    that names one call site misses the next, and there are already several.

    Only handlers that were not there before are removed, so a handler
    configured by the harness itself survives.
    """
    import logging

    watched = ("umi.common.data.data_buffer", "real_scripts.data_processing.align_data")
    before = {name: list(logging.getLogger(name).handlers) for name in watched}
    yield
    for name, original in before.items():
        logger = logging.getLogger(name)
        for added in [h for h in logger.handlers if h not in original]:
            logger.removeHandler(added)
            added.close()
