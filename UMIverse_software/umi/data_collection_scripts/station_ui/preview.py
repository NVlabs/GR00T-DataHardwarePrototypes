# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The configuration page's live camera preview: one at a time, never left open.

Clicking an OAK, iPhone or USB camera id in the detected-devices table opens a
pop-up with that camera's live picture, so an operator tells look-alike devices
apart by what they see instead of by unplugging cables. The picture comes from
`station_ui.preview_worker`, a capped child writing its newest frame to a file
named here; `routes/preview.py` streams that file into the pop-up.

An OAK open is exclusive, so a preview competes with everything else that opens
cameras. The rules:

* one preview at a time -- opening another stops the first;
* it never starts under a recording, a device scan or the camera check
  (`routes/preview.py` refuses);
* a scan, the camera check and a recording each stop it before they open
  anything, the same release `calibration.stop_check` gives the camera check;
* it ends when the pop-up closes, and after `TIMEOUT_S` regardless -- a tab shut
  without a word must not hold a camera for the rest of the shift.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time
import uuid
from dataclasses import dataclass

from station_ui import devices as device_state
from station_ui import jobs
from station_ui.errors import UnknownJob

__all__ = [
    "KINDS",
    "RELEASE_WAIT_S",
    "TIMEOUT_S",
    "PreviewState",
    "current_job",
    "preview_argv",
    "running",
    "start",
    "status",
    "stop",
    "stop_job",
]

# What the table can preview: the kinds with a picture. A gripper has none.
KINDS = ("oak", "iphone", "uvc")
TIMEOUT_S = 120.0
# Wait for a stopped preview to release its camera before reopening it.
RELEASE_WAIT_S = 6.0
WORKER_ERROR = "[preview] error: "


@dataclass
class PreviewState:
    """The preview this console last started, and where its frames land."""

    job_id: str | None = None
    frame_path: str = ""


_ATTACH = threading.Lock()
# Serialize stop-then-start so concurrent requests cannot launch previews
# against the same board.
_START = threading.Lock()


def state_for(app) -> PreviewState:
    with _ATTACH:
        state = getattr(app.state, "preview", None)
        if state is None:
            state = PreviewState()
            app.state.preview = state
        return state


def current_job(app):
    """The latest preview's job, finished or not; None if there never was one.

    Reads `app.state` directly so asking does not attach anything to an app
    that has never previewed, the same courtesy `calibration.check_running`
    extends.
    """
    state = getattr(app.state, "preview", None)
    if state is None or not state.job_id:
        return None
    try:
        return device_state.registry_for(app).get(state.job_id)
    except UnknownJob:
        return None


def running(app) -> bool:
    job = current_job(app)
    return job is not None and not job.finished


def stop_job(app, job_id) -> None:
    """Cancel one preview by id. Idempotent; an unknown id is a no-op."""
    try:
        job = device_state.registry_for(app).get(job_id)
    except UnknownJob:
        return
    if job.kind == "preview" and not job.finished:
        jobs.cancel(device_state.registry_for(app), job_id)


def stop(app, *, wait: float = 0.0) -> None:
    """Release the camera the preview holds, waiting up to `wait` seconds for
    the child to actually close it. Run off the event loop when waiting."""
    job = current_job(app)
    if job is None or job.finished:
        return
    stop_job(app, job.id)
    deadline = time.monotonic() + wait
    while not job.finished and time.monotonic() < deadline:
        time.sleep(0.02)


def preview_argv(kind: str, device_id: str, frame_path: str) -> list[str]:
    """The command one preview runs. A function so tests can swap in a fake
    child: no test may open a camera.

    `--opt=value` rather than two words, so a device id that happened to start
    with a dash could never be read as an option.
    """
    return [
        sys.executable,
        "-m",
        "station_ui.preview_worker",
        f"--kind={kind}",
        f"--device={device_id}",
        f"--out={frame_path}",
    ]


def start(app, kind: str, device_id: str):
    """Stop any preview, then start one for `device_id`. Returns the job.

    Blocks up to `RELEASE_WAIT_S` while the previous preview lets go of its
    camera, so call it off the event loop. The caller has already refused the
    cases the module docstring lists.
    """
    with _START:
        stop(app, wait=RELEASE_WAIT_S)
        directory = pathlib.Path(app.state.settings.state_dir) / "preview"
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.iterdir():
            stale.unlink(missing_ok=True)
        frame = directory / f"{uuid.uuid4().hex[:12]}.jpg"
        job = jobs.start(
            device_state.registry_for(app),
            preview_argv(kind, device_id, str(frame)),
            cwd=device_state.REPO_ROOT,
            timeout=TIMEOUT_S,
            label=f"camera preview {device_id}",
            kind="preview",
        )
        state = state_for(app)
        state.job_id = job.id
        state.frame_path = str(frame)
        return job


def status(app, job_id) -> dict:
    """What the pop-up shows under the picture: `{"running", "streaming", "message"}`."""
    try:
        job = device_state.registry_for(app).get(job_id)
    except UnknownJob:
        return {
            "running": False,
            "streaming": False,
            "message": "This preview has ended. Click the ID again to reopen it.",
        }
    if not job.finished:
        state = state_for(app)
        streaming = state.job_id == job_id and pathlib.Path(state.frame_path).is_file()
        return {"running": True, "streaming": streaming, "message": ""}
    return {"running": False, "streaming": False, "message": _ended(job)}


def _ended(job) -> str:
    for line in reversed(job.lines()):
        if line.startswith(WORKER_ERROR):
            reason = line[len(WORKER_ERROR) :]
            return reason[:1].upper() + reason[1:] + "."
    if (job.error or "").startswith("timed out"):
        minutes = round(TIMEOUT_S / 60)
        return (
            f"The preview closed after {minutes} minutes so the camera is free "
            "again. Click the ID to reopen it."
        )
    if job.state == jobs.CANCELLED:
        return "The preview was stopped so the camera could be used elsewhere."
    return "The preview ended."
