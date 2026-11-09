# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared by every Station Console test.

They keep tests away from the real station (its log, its state directory, the
user's home directory and real terminal windows) and reset process-wide state
that would otherwise carry over from one test to the next.
"""

from __future__ import annotations

import pytest

from station_ui import crash, settings


@pytest.fixture(scope="session", autouse=True)
def station_log_dir(tmp_path_factory):
    """Point the station log at a temporary directory for the whole session.

    Session-scoped because `create_app` attaches the log handler once per
    process, when the first app is built; a function-scoped fixture would come
    too late for every later test.
    """
    directory = tmp_path_factory.mktemp("station-log")
    patch = pytest.MonkeyPatch()
    patch.setenv(settings.LOG_DIR_ENV, str(directory))
    yield directory
    patch.undo()


@pytest.fixture(autouse=True)
def station_state_dir(tmp_path_factory, monkeypatch):
    """Give every test its own station state directory.

    Without it, `create_app` would restore the session last applied in the
    real console of whoever runs the suite, and tests that expect an empty
    console would fail on that machine. Function-scoped because `Settings`
    reads the directory each time it is constructed.
    """
    directory = tmp_path_factory.mktemp("station-state")
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(directory))
    return directory


@pytest.fixture(autouse=True)
def station_home(tmp_path, monkeypatch):
    """Give every test its own `$HOME`.

    `build_data_dir` returns a path under `~`, so a test that applies a config
    and records an episode would otherwise write into the real data directory
    of whoever runs the suite. A seal marker written there would stop a live
    recorder from using that session. Autouse, because any test that drives
    the pages can trigger this; tests that set `HOME` themselves still win.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def no_terminal_window(monkeypatch):
    """Point gripper calibration at a terminal that does not exist.

    Otherwise a test that drives `/calibration/calibrate-gripper` without
    stubbing the launcher would open a real terminal window running the
    calibration in gnome-terminal. With no terminal the job fails to start, as
    it does on a CI runner. Tests that set
    `TERMINAL_ARGV` themselves still win.
    """
    from station_ui import calibration

    monkeypatch.setattr(calibration, "TERMINAL_ARGV", ("/nonexistent/terminal", "--"))


@pytest.fixture(autouse=True)
def _forget_crashes():
    crash.reset()
    yield
    crash.reset()


# Disable the OAK discovery gate by default for tests without scans.
# Tests of the gate opt in with pytest.mark.oak_gate.
@pytest.fixture(autouse=True)
def _oak_gate_stood_down(request, monkeypatch):
    if request.node.get_closest_marker("oak_gate") is not None:
        return
    from station_ui.routes import config as config_routes

    monkeypatch.setattr(
        config_routes, "_unscanned_oak_fields", lambda *a, **k: ({}, "")
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "oak_gate: run the real Apply gate for unscanned OAK cameras"
    )
