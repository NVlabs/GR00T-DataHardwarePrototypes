# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guided camera identification and camera-role verification."""

import asyncio
import time

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from station_ui import (
    calibration,
    config_form,
    devices,
    gripper_pairing,
    identification,
    markers,
    preview,
    recording,
    tokens,
)
from station_ui import session as session_state
from station_ui.camera_access import serialize_start
from station_ui.errors import RecordingInProgress, StaleOwner
from station_ui.rig import is_owner, require_owner
from station_ui.routes import config

router = APIRouter()
PAGE = "/config/identify"


def owner(request):
    try:
        require_owner(request.app.state.rig, tokens.token_from(request))
        recording.require_idle(request.app)
    except (StaleOwner, RecordingInProgress) as exc:
        raise HTTPException(409, str(exc)) from exc


def session(request, session_id):
    state = identification.current(request.app)
    if state is None or state.id != session_id:
        raise HTTPException(
            404, "This camera identification has ended. Open Identify Cameras again."
        )
    return state


def slots_for(requirements):
    slots = {}
    for side in sorted(requirements.gripper_arms):
        blocks = [
            block
            for block in requirements.blocks
            if any(name.startswith(side + "_") for name in block.stream_names)
        ]
        grouped = {}
        for block in blocks:
            if block.device_kind not in ("oak", "iphone", "uvc"):
                continue
            if block.device_kind in grouped:
                raise ValueError(
                    f"More than one {block.device_kind} camera slot on {side}; select these cameras manually."
                )
            grouped[block.device_kind] = block
        if grouped:
            slots[side] = grouped
    if not slots:
        raise ValueError("Choose a setup before identifying cameras.")
    return slots


@router.post(PAGE + "/refresh")
@serialize_start
async def refresh(request: Request):
    """Release live identification workers before the existing probe rescan."""
    owner(request)
    state = identification.current(request.app)
    if state and not state.released():
        if state.token != tokens.token_from(request):
            raise HTTPException(409, "This identification belongs to another browser.")
        state.stop(request.app)
        deadline = time.monotonic() + identification.RELEASE_SECONDS + 3
        while not state.released() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if not state.released():
            raise HTTPException(
                409, "Cameras are still releasing. Wait, then refresh again."
            )
    owner(request)
    if calibration.check_running(request.app) or preview.running(request.app):
        raise HTTPException(
            409, "Close the other camera check or preview before refreshing devices."
        )
    await config._absorbed(request, request.app.state.rig.draft)
    job, _ = config._start_scan(request)
    return {"scan_id": job.id}


@router.get(PAGE + "/inventory")
async def inventory(request: Request):
    job = devices.current_job(request.app)
    found, error = devices.outcome_for(job)
    return {
        "scan_id": job.id if job else None,
        "running": job is not None and not job.finished,
        "error": error,
        "count": len(found) if found is not None else None,
    }


@router.post(PAGE + "/start")
@serialize_start
async def start(request: Request):
    await request.form()  # cache the body before checking ownership and opening SDKs
    owner(request)
    if identification.running(request.app):
        state = identification.current(request.app)
        return JSONResponse(
            {"error": identification.MESSAGE, "id": state.id}, status_code=409
        )
    scan = devices.current_job(request.app)
    if scan is not None and not scan.finished:
        raise HTTPException(
            409, "Wait for Refresh devices to finish before identifying cameras."
        )
    if calibration.check_running(request.app):
        raise HTTPException(409, devices.CHECK_HOLDS_THE_CAMERAS)
    if preview.running(request.app):
        raise HTTPException(
            409, "Close the existing camera preview before identifying cameras."
        )
    found, error = devices.outcome_for(scan)
    if found is None or error:
        raise HTTPException(
            409, "Refresh devices successfully before identifying cameras."
        )
    # Preserve half-filled fields before opening the dialog. No config is written.
    _, requirements = await config._absorbed(request, request.app.state.rig.draft)
    try:
        slots = slots_for(requirements)
        kinds = {kind for group in slots.values() for kind in group}
        camera_records = [record for record in found if record.get("kind") in kinds]
        state = identification.start(
            request.app,
            tokens.token_from(request),
            camera_records,
            session_state.device_fingerprint(request.app.state.rig.draft),
            inventory=found,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    state.slots = slots
    return {**state.payload(), "sides": list(slots)}


@router.get(PAGE + "/{session_id}/status")
async def status(request: Request, session_id: str):
    state = session(request, session_id)
    return {**state.payload(), "sides": list(getattr(state, "slots", {}))}


@router.get(PAGE + "/{session_id}/frame/{job_id}")
async def frame(request: Request, session_id: str, job_id: str):
    state = session(request, session_id)
    camera = next((camera for camera in state.cameras if camera.job.id == job_id), None)
    if camera is None or state.closing or camera.job.finished:
        return Response(status_code=404)
    try:
        content = await run_in_threadpool(camera.frame.read_bytes)
    except FileNotFoundError:
        return Response(status_code=204)
    return Response(
        content, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


@router.get(PAGE + "/marker/{side}.svg")
async def marker(side: str):
    if side not in ("left", "right"):
        raise HTTPException(404, "Unknown side")
    return Response(markers.marker_svg(side), media_type="image/svg+xml")


@router.post(PAGE + "/{session_id}/scan")
@serialize_start
async def scan_side(request: Request, session_id: str, side: str = Form(...)):
    owner(request)
    state = session(request, session_id)
    if state.token != tokens.token_from(request) or state.closing or state.scanning:
        raise HTTPException(
            409, "Identification is busy or belongs to another browser."
        )
    draft = request.app.state.rig.draft
    if state.fingerprint != session_state.device_fingerprint(draft):
        state.stop(request.app)
        raise HTTPException(
            409, "The configuration changed. Close and identify the cameras again."
        )
    if side not in state.slots:
        raise HTTPException(400, "Unknown gripper side")
    if side == "right" and "left" in state.slots and "left" not in state.completed:
        raise HTTPException(
            409, "Identify the left cameras before scanning the right side."
        )
    cameras = state.payload()["cameras"]
    if any(not camera["ready"] or camera["error"] for camera in cameras):
        raise HTTPException(
            409,
            "Wait for every camera to show a live preview. Resolve camera errors before scanning.",
        )
    state.scanning = True
    try:
        assignments, errors = await run_in_threadpool(
            state.scan, side, state.slots[side]
        )
        # A tab can lose ownership while the worker observes frames.
        owner(request)
        if (
            state.closing
            or state.token != tokens.token_from(request)
            or state.fingerprint != session_state.device_fingerprint(draft)
        ):
            state.stop(request.app)
            raise HTTPException(
                409,
                "The configuration or camera session changed. Existing selections were kept.",
            )
        if errors:
            return JSONResponse({"errors": errors, "assignments": {}}, status_code=422)
        fields = {}
        cleared = set()
        cleared_streams = set()
        confirmed_streams = {}
        for kind, name in assignments.items():
            block = state.slots[side][kind]
            for other, blocks in state.slots.items():
                if other == side or kind not in blocks:
                    continue
                other_block = blocks[kind]
                if any(
                    draft.required_cameras.get(stream) == name
                    for stream in other_block.stream_names
                ):
                    for stream in other_block.stream_names:
                        draft.required_cameras.pop(stream, None)
                        cleared_streams.add(stream)
                    fields[config_form.camera_field(other_block.key)] = ""
                    state.completed.discard(other)
                    cleared.add(other)
            for stream in block.stream_names:
                draft.required_cameras[stream] = name
                confirmed_streams[stream] = name
            fields[config_form.camera_field(block.key)] = name
        gripper, gripper_warning = gripper_pairing.gripper_for_cameras(
            assignments, state.inventory
        )
        if gripper:
            for other, selected in list(draft.grippers.items()):
                if other != side and selected == gripper:
                    draft.grippers.pop(other)
                    fields[config_form.gripper_field(other)] = ""
            draft.grippers[side] = gripper
            fields[config_form.gripper_field(side)] = gripper
        state.assignments.update(fields)
        state.completed.add(side)
        state.fingerprint = session_state.device_fingerprint(draft)
        identification.remember_verification(
            request.app, confirmed_streams, cleared_streams
        )
        messages = [gripper_warning] if gripper_warning else []
        if cleared:
            messages.append(
                "Rescan "
                + ", ".join(sorted(cleared))
                + " because the same cameras were previously selected there."
            )
        return {
            "assignments": fields,
            "completed": sorted(state.completed),
            "message": " ".join(messages),
        }
    finally:
        state.scanning = False


@router.post(PAGE + "/{session_id}/stop")
async def stop(request: Request, session_id: str):
    state = session(request, session_id)
    if state.token != tokens.token_from(request) and not is_owner(
        request.app.state.rig, tokens.token_from(request)
    ):
        raise HTTPException(409, "This identification belongs to another browser.")
    state.stop(request.app)
    return state.payload()
