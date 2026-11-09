# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/config/preview` -- a live picture from one detected camera, in a pop-up.

The detected-devices table on `/config` makes each OAK, iPhone and USB camera
id a button; `static/device_preview.js` posts it here and shows what comes
back. The camera work is `station_ui.preview_worker`, started and stopped by
`station_ui.preview` (which also holds the rules for when a preview may run).
This module owns only the HTTP shape: JSON for the pop-up's script, and the
picture itself as a multipart JPEG stream an `<img>` plays without any script
at all.
"""

from __future__ import annotations

import asyncio
import pathlib

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from station_ui import calibration as cal
from station_ui import devices as device_state
from station_ui import identification, preview, recording, tokens
from station_ui.camera_access import serialize_start
from station_ui.errors import RecordingInProgress, StaleOwner
from station_ui.rig import require_owner

router = APIRouter()

PAGE = "/config/preview"
BAD_REQUEST = 400
NOT_FOUND = 404
CONFLICT = 409
NO_CONTENT = 204

BOUNDARY = "preview-frame"
# How often the stream looks for a newer picture. Faster than the worker
# writes one, so a frame is never more than a poll late.
FRAME_POLL_S = 0.05

REFRESHING = "Devices are being refreshed. Open the preview when the refresh finishes."
NOT_SCANNED = (
    "That camera is not in the last device scan. Press Refresh devices and try again."
)


def _refused(status_code: int, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code)


def _scanned(found, kind: str, device: str):
    """The last scan's record for the id the operator clicked, or None.

    Only a device the scan found can be previewed: the click comes from the
    scan's own table, and this is what turns a registry name back into the
    raw id the camera is opened by.
    """
    for record in found or []:
        if record.get("kind") != kind:
            continue
        if device in (device_state.identifier_for(record), record.get("device_id")):
            return record
    return None


@router.post(f"{PAGE}/start")
@serialize_start
async def start_preview(request: Request, kind: str = Form(""), device: str = Form("")):
    app = request.app
    try:
        require_owner(app.state.rig, tokens.token_from(request))
    except StaleOwner as exc:
        return _refused(CONFLICT, str(exc))
    if identification.running(app):
        return _refused(CONFLICT, identification.MESSAGE)
    try:
        recording.require_idle(app)
    except RecordingInProgress as exc:
        return _refused(CONFLICT, str(exc))
    # Refuse preview while verification owns cameras; only the operator can
    # stop that active check.
    if cal.check_running(app):
        return _refused(CONFLICT, device_state.CHECK_HOLDS_THE_CAMERAS)
    scan = device_state.current_job(app)
    if scan is not None and not scan.finished:
        return _refused(CONFLICT, REFRESHING)
    if kind not in preview.KINDS:
        return _refused(
            BAD_REQUEST, "Only OAK cameras, iPhones and USB cameras have a preview."
        )

    found, _ = device_state.outcome_for(scan)
    record = _scanned(found, kind, device.strip())
    if record is None:
        return _refused(BAD_REQUEST, NOT_SCANNED)
    if record.get("state") != device_state.AVAILABLE:
        return _refused(
            CONFLICT,
            f"The last scan could not open {device}. Free it and press Refresh devices.",
        )

    job = await run_in_threadpool(preview.start, app, kind, record["device_id"])
    return JSONResponse(
        {
            "job_id": job.id,
            "device": device,
            "stream_url": f"{PAGE}/stream/{job.id}",
            "status_url": f"{PAGE}/status/{job.id}",
            "stop_url": f"{PAGE}/stop/{job.id}",
        }
    )


async def frames(app, job, path: pathlib.Path, is_disconnected=None):
    """The preview's pictures as multipart parts, until the preview ends.

    The `finally` is the release: when the pop-up stops reading -- closed,
    reloaded, the tab shut -- the generator ends, and the camera goes with it.
    Under uvicorn a dropped connection only surfaces when a send fails, and an
    OAK takes seconds to send its first picture (one that never sends, never),
    so `is_disconnected` is also asked on every poll. The wall-clock cap in
    `preview` covers the case where even that never arrives.
    """
    sent = None
    try:
        while not job.finished:
            if is_disconnected is not None and await is_disconnected():
                break
            try:
                stat = path.stat()
                stamp = (stat.st_mtime_ns, stat.st_size)
                if stamp != sent:
                    data = path.read_bytes()
                    sent = stamp
                    yield (
                        (
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(data)}\r\n\r\n"
                        ).encode()
                        + data
                        + b"\r\n"
                    )
            except FileNotFoundError:
                pass  # no picture yet, or the worker is tidying up on exit
            await asyncio.sleep(FRAME_POLL_S)
    finally:
        preview.stop_job(app, job.id)


@router.get(f"{PAGE}/stream/{{job_id}}")
async def stream_preview(request: Request, job_id: str):
    state = preview.state_for(request.app)
    job = preview.current_job(request.app)
    # Serve only the current preview so stale popup URLs cannot reacquire
    # a camera now used by another operation.
    if job is None or job.id != job_id or job.finished:
        return Response(status_code=NOT_FOUND)
    return StreamingResponse(
        frames(
            request.app,
            job,
            pathlib.Path(state.frame_path),
            is_disconnected=request.is_disconnected,
        ),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


@router.get(f"{PAGE}/status/{{job_id}}")
async def preview_status(request: Request, job_id: str):
    return JSONResponse(preview.status(request.app, job_id))


@router.post(f"{PAGE}/stop/{{job_id}}")
async def stop_preview(request: Request, job_id: str):
    """Release the camera. Deliberately not owner-gated: letting go of a camera
    can never hurt, and a pop-up whose tab lost control must still close."""
    preview.stop_job(request.app, job_id)
    return Response(status_code=NO_CONTENT)
