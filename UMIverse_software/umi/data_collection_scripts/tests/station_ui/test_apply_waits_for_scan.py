# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply does not write an OAK's picture size it has not read from the scan.

Without a scan result behind the head camera, Apply would fall back to the
wrist default (640x400), which crops 17% of the head's 4:3 IMX378 -- and a
warning banner alone is easy to save past. This file pins two rules:

  1. Apply refuses while the scan is still running, and refuses an OAK the
     finished scan did not report a module for. Nothing is written.
  2. The head camera's fallback, for paths with no page to refuse on, is the
     head size (640x480), not the wrist size.

No test here opens hardware.
"""

from __future__ import annotations

import json
import sys
import time

import pytest

yaml = pytest.importorskip("yaml")

import test_unavailable_device_gate as gate  # noqa: E402

from station_ui import devices as devices_core  # noqa: E402
from station_ui import scan_worker  # noqa: E402
from station_ui.core.device_scan import (  # noqa: E402
    DEFAULT_STREAM_RESOLUTION,
    HEAD_STREAM_RESOLUTION,
    DeviceScanner,
    resolution_for,
)

pytestmark = pytest.mark.oak_gate

BAD_REQUEST = 400
registry = gate.registry  # the autouse fixtures of the gate suite
repo = gate.repo
client = gate.client

FULL_SCAN = [
    gate.oak_found(gate.OAK_R_ID, gate.OAK_R),
    gate.oak_found(gate.OAK_L_ID, gate.OAK_L),
    gate.HEAD_FOUND,
]


def slow_scan_argv(found, hold_s: float):
    """A child that reports, then keeps running: the scan 'still in flight'."""
    payload = scan_worker.RESULT_PREFIX + json.dumps({"devices": list(found)})
    source = (
        "import sys, time\n"
        "print('[scan] starting', flush=True)\n"
        f"time.sleep({hold_s})\n"
        f"print({payload!r}, flush=True)\n"
        "sys.exit(0)\n"
    )
    return [sys.executable, "-c", source]


def test_apply_is_refused_while_the_scan_is_still_running(client, repo, monkeypatch):
    monkeypatch.setattr(
        devices_core, "scan_argv", lambda: slow_scan_argv(FULL_SCAN, 6.0)
    )
    client.post("/config/devices/scan")
    job = devices_core.current_job(client.app)
    assert job is not None and not job.finished

    response = client.post("/config/apply", data=gate.handcrafted())

    assert response.status_code == BAD_REQUEST, response.text[:400]
    assert "still running" in response.text
    assert "Wait for the device scan to finish" in response.text
    assert not gate.generated(repo).exists()
    # And once it has finished, the same post goes through.
    end = time.monotonic() + gate.DEADLINE_S
    while not job.finished and time.monotonic() < end:
        time.sleep(0.05)
    assert job.finished
    assert client.post("/config/apply", data=gate.handcrafted()).status_code == 200


def test_a_finished_scan_that_missed_the_head_refuses_the_head_row_only(
    client, repo, monkeypatch
):
    """The 'head only missed' case: wrists read, head not reported (held by a
    recorder still closing, or re-plugged after the page opened)."""
    gate.run_scan(client, monkeypatch, FULL_SCAN[:2])

    response = client.post("/config/apply", data=gate.handcrafted())

    assert response.status_code == BAD_REQUEST, response.text[:400]
    assert response.text.count("Not seen by the device scan") == 1
    assert not gate.generated(repo).exists()


def test_the_head_falls_back_to_its_own_size_and_the_wrists_to_theirs():
    """For the headless assembly paths: no scan, two different right answers."""
    empty = DeviceScanner()
    head, head_warning = resolution_for(
        "oak-head", "CAM_A", empty, default=HEAD_STREAM_RESOLUTION
    )
    wrist, wrist_warning = resolution_for("oak-FFC-01", "CAM_A", empty)
    assert head == [640, 480] and head_warning and "640, 480" in head_warning
    assert wrist == list(DEFAULT_STREAM_RESOLUTION) == [640, 400] and wrist_warning
    # A scanned head still comes from the module, not the default.
    scanned = DeviceScanner.from_records([gate.HEAD_FOUND])
    assert resolution_for("oak-head", "CAM_A", scanned, default=[1, 1]) == (
        [640, 480],
        None,
    )
