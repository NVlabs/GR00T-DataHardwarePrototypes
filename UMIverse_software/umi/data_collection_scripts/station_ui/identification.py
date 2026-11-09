# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live marker identification for every UMI camera kind, without saving configs.

Only an unambiguous, complete set of fresh observations changes the draft.
Workers use the recorder's camera SDKs through the existing preview sources.
"""

import json
import os
import pathlib
import sys
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field

from station_ui import camera_roles, devices, jobs
from station_ui.core.config_assembler import source_device
from station_ui.identify_worker import PREFIX

STARTUP_SECONDS = 50.0
SCAN_SECONDS = 5.0
MIN_HITS = 3
IDLE_SECONDS = 45.0
MAX_SECONDS = 300.0
RELEASE_SECONDS = 15.0
MESSAGE = (
    "Camera identification is open. Close it and wait for the cameras to release first."
)


def forget_verification(app):
    app.state.identified_cameras = None


def remember_verification(app, streams, cleared=()):
    """Keep marker evidence per assignment, independently of worker lifetime."""
    draft = app.state.rig.draft
    proof = getattr(app.state, "identified_cameras", None)
    if proof is None or proof["preset"] != draft.preset_id:
        proof = {"preset": draft.preset_id, "streams": {}}
        app.state.identified_cameras = proof
    for stream in cleared:
        proof["streams"].pop(stream, None)
    proof["streams"].update(streams)


def verified_roles(app, config=None):
    """Evidence applies only to the same preset, stream and selected camera.

    When checking recording readiness, also compare the generated file: draft
    evidence must never vouch for a different device in a saved config.
    """
    draft = app.state.rig.draft
    proof = getattr(app.state, "identified_cameras", None)
    if not proof or proof["preset"] != draft.preset_id:
        return set()
    configured = {}
    if config is not None:
        for source in config.get("video_sources") or []:
            names = [source.get("name")] + [
                stream.get("name") for stream in source.get("streams") or []
            ]
            configured.update({name: source_device(source) for name in names if name})
    return {
        role
        for role, names in camera_roles.ROLE_SOURCE_NAMES.items()
        if any(
            (value := proof["streams"].get(name))
            and draft.required_cameras.get(name) == value
            and (config is None or configured.get(name) == value)
            for name in names
        )
    }


@dataclass
class Camera:
    kind: str
    device_id: str
    name: str
    job: object
    frame: pathlib.Path


def events(camera):
    result = []
    for line in camera.job.lines():
        if line.startswith(PREFIX):
            try:
                value = json.loads(line[len(PREFIX) :])
                if isinstance(value, dict):
                    result.append(value)
            except (ValueError, TypeError):
                continue
    return result


def observations(camera):
    rows = events(camera)
    error = next((str(row["error"]) for row in reversed(rows) if row.get("error")), "")
    latest = next((row for row in reversed(rows) if "seq" in row), {})
    ready = any(row.get("ready") for row in rows)
    if camera.job.finished and not error:
        error = (
            camera.job.error or "The camera worker stopped. Close and identify again."
        )
    return rows, latest, ready, error


@dataclass
class Identification:
    id: str
    token: str
    fingerprint: object
    cameras: list = field(default_factory=list)
    slots: dict = field(default_factory=dict)
    inventory: list = field(default_factory=list)
    closing: bool = False
    scanning: bool = False
    last_seen: float = field(default_factory=time.monotonic)
    started: float = field(default_factory=time.monotonic)
    completed: set = field(default_factory=set)
    assignments: dict = field(default_factory=dict)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def stop(self, app):
        self.closing = True
        self.cancelled.set()
        registry = devices.registry_for(app)
        for camera in self.cameras:
            if not camera.job.finished:
                jobs.cancel(registry, camera.job.id)

    def released(self):
        return self.closing and all(camera.job.finished for camera in self.cameras)

    def payload(self):
        self.last_seen = time.monotonic()
        cameras = []
        for camera in self.cameras:
            _, latest, ready, error = observations(camera)
            if not ready and time.monotonic() - self.started > STARTUP_SECONDS:
                error = (
                    error or "Camera startup timed out. Close and check the connection."
                )
            cameras.append(
                {
                    "kind": camera.kind,
                    "name": camera.name,
                    "device_id": camera.device_id,
                    "ready": ready,
                    "error": error,
                    "sees": latest.get("sides", [])
                    if time.monotonic() - latest.get("at", 0) < 1.5
                    else [],
                    "frame_url": f"/config/identify/{self.id}/frame/{camera.job.id}",
                }
            )
        return {
            "id": self.id,
            "cameras": cameras,
            "scanning": self.scanning,
            "closing": self.closing,
            "released": self.released(),
            "completed": sorted(self.completed),
            "assignments": dict(self.assignments),
        }

    def scan(self, side, kinds):
        """No state mutation here: the route rechecks ownership before committing."""
        started = time.monotonic()
        while (
            not self.cancelled.wait(0.08) and time.monotonic() - started < SCAN_SECONDS
        ):
            pass
        errors, assignments = [], {}
        if self.cancelled.is_set():
            return {}, ["Identification was cancelled. Existing selections were kept."]
        for kind in kinds:
            matches = []
            for camera in self.cameras:
                if camera.kind != kind:
                    continue
                rows, _, ready, error = observations(camera)
                if not ready or error:
                    errors.append(
                        f"{camera.name}: {error or 'the camera is not ready'}"
                    )
                    continue
                fresh_frames = {
                    row["seq"]
                    for row in rows
                    if row.get("at", 0) >= started and "seq" in row
                }
                if len(fresh_frames) < MIN_HITS:
                    errors.append(
                        f"{camera.name} did not deliver enough fresh frames. Check its preview and scan again."
                    )
                votes = {
                    row["seq"]
                    for row in rows
                    if row.get("at", 0) >= started
                    and side in row.get("sides", [])
                    and "seq" in row
                }
                if len(votes) >= MIN_HITS:
                    matches.append(camera.name)
            if len(matches) == 1:
                assignments[kind] = matches[0]
            elif matches:
                errors.append(
                    f"Multiple {kind} cameras see the {side} marker. Point the other cameras away and scan again."
                )
            else:
                errors.append(
                    f"No {kind} camera clearly saw the {side} marker. Aim it at the marker and scan again."
                )
        return ({} if errors else assignments), errors


_ATTACH = threading.Lock()


def current(app):
    return getattr(app.state, "camera_identification", None)


def running(app):
    state = current(app)
    return state is not None and not state.released()


def start(app, token, found, fingerprint, inventory=None):
    with _ATTACH:
        if running(app):
            raise ValueError(MESSAGE)
        state = Identification(uuid.uuid4().hex, token, fingerprint)
        state.inventory = deepcopy(inventory if inventory is not None else found)
        app.state.camera_identification = state  # reserve before spawning any SDK
        directory = (
            pathlib.Path(app.state.settings.state_dir) / "identification" / state.id
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            seen = set()
            for record in found:
                kind, device_id = record.get("kind"), str(record.get("device_id") or "")
                if (
                    kind not in ("oak", "iphone", "uvc")
                    or not device_id
                    or (kind, device_id) in seen
                ):
                    continue
                seen.add((kind, device_id))
                if record.get("state") != devices.AVAILABLE:
                    raise ValueError(
                        f"{devices.identifier_for(record)} is unavailable. Free it and Refresh devices first."
                    )
                frame = directory / f"{len(state.cameras)}.jpg"
                job = jobs.start(
                    devices.registry_for(app),
                    [
                        sys.executable,
                        "-m",
                        "station_ui.identify_worker",
                        f"--kind={kind}",
                        f"--device={device_id}",
                        f"--out={frame}",
                        f"--parent-pid={os.getpid()}",
                    ],
                    cwd=app.state.settings.repo_root,
                    timeout=MAX_SECONDS,
                    term_grace=RELEASE_SECONDS,
                    kind="identification",
                    label=f"Identify {kind} {device_id}",
                )
                state.cameras.append(
                    Camera(kind, device_id, devices.identifier_for(record), job, frame)
                )
            if not state.cameras:
                raise ValueError(
                    "No cameras found. Connect cameras and Refresh devices first."
                )
        except Exception:
            state.stop(app)
            raise

        def expire():
            while not state.closing:
                if (
                    time.monotonic() - state.last_seen > IDLE_SECONDS
                    or time.monotonic() - state.started > MAX_SECONDS
                ):
                    state.stop(app)
                    break
                time.sleep(1)

        threading.Thread(
            target=expire, daemon=True, name="camera-identification-expiry"
        ).start()
        return state
