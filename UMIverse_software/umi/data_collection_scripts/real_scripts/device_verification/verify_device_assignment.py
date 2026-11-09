# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that each camera sees the marker its role is supposed to see.

Roles: left_tracking (iPhone), left_umi (OAK), left_trigger, right_tracking,
right_umi, right_trigger, head (iPhone), and head_umi (OAK). The same
LEFT/RIGHT/HEAD marker is shared by every camera in that position. The bottom table
OAK is recorded but not checked, so it has no role here.

``--watch`` (used by the Station Console) opens every camera the config
declares, all at once and with one open per physical device, draws nothing,
and prints one prefixed JSON line per *change* in what a camera decodes. It
never learns which marker is actually on screen: it says "left_umi decoded
UMI_VERIFY_RIGHT" and lets ``station_ui/calibration.py`` call that a
cross-wired camera. The marker itself is served by the browser
(``/calibration/check/qr/{side}.svg``), so the operator carries the page to the
rig instead of walking back to the station monitor. Runs until SIGTERM; the
handler installed in ``main`` is what lets the ``finally`` release every camera.

Supports ``oak_camera`` / ``uvc_camera`` entries with a top-level ``name``, and
``oak_device`` entries (OAK-FFC-3P) where logical names live under ``streams``.
Which source name carries which role is ``station_ui.camera_roles``, shared with
the server so the two cannot disagree about what gets verified.
"""

import argparse
import copy
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from station_ui.camera_roles import HEAD_ROLES, ROLE_SOURCE_NAMES
from station_ui.markers import MARKER_IDS
from umi.common.hardware_sdk.camera.iphone_camera import IphoneCamera
from umi.common.hardware_sdk.camera.oak_camera import OAKCamera
from umi.common.hardware_sdk.camera.oak_device import OAKDevice
from umi.common.hardware_sdk.camera.uvc_camera import UVCCamera

PAYLOAD_PREFIX = "UMI_VERIFY_"

# Read ArUco markers for their large modules, which are easier to resolve
# on screens through fisheye lenses. Map marker IDs through the same
# station_ui.markers table used for display.
_ARUCO_SIDE_BY_ID = {marker_id: side.upper() for side, marker_id in MARKER_IDS.items()}


def _aruco_detector():
    """The detector, built once. None where this OpenCV has no `aruco`."""
    if not hasattr(cv2, "aruco"):
        return None
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    # Refine softened screen edges and tolerate curved marker outlines from
    # fisheye projection.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.polygonalApproxAccuracyRate = 0.08
    return cv2.aruco.ArucoDetector(dictionary, params)


_DETECTOR = None


def decode_marker(frame):
    """The payload this frame shows, or None.

    Tries the full frame and then its centre. Barrel distortion is worst at the
    edge and near zero on the optical axis, so a marker that will not resolve
    across a 200-degree field often resolves cleanly once the periphery -- which
    is most of the pixels and none of the information -- is cropped away and what
    is left is scaled back up.
    """
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _aruco_detector() or False

    for candidate in _frame_candidates(frame):
        if _DETECTOR:
            try:
                _corners, ids, _rejected = _DETECTOR.detectMarkers(candidate)
            except Exception:  # noqa: BLE001 - a detector fault is not a verdict
                ids = None
            if ids is not None and len(ids):
                side = _ARUCO_SIDE_BY_ID.get(int(ids.flatten()[0]))
                if side:
                    return f"{PAYLOAD_PREFIX}{side}"
    return None


def _frame_candidates(frame):
    """The frame, then its upscaled centre. Cheap, and the second one is what
    reads a marker the lens has bent out of recognition at the edge."""
    yield frame
    try:
        height, width = frame.shape[:2]
        if min(height, width) < 80:
            return
        top, left = height // 4, width // 4
        centre = frame[top : top + height // 2, left : left + width // 2]
        yield cv2.resize(centre, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    except Exception:  # noqa: BLE001 - a crop that fails is one fewer attempt
        return


class OAKDeviceStreamVerifyCamera:
    """Single stream from an :class:`OAKDevice` for marker verification (OAK-FFC / multi-stream YAML)."""

    def __init__(
        self,
        *,
        oak_device_name: Optional[str] = None,
        oak_device_id: Optional[str] = None,
        fps: int = 30,
        latency: float = 0.0,
        stream: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._oak_device_name = oak_device_name
        self._oak_device_id = oak_device_id
        self._fps = fps
        self._latency = latency
        self._stream = stream or {}
        self._device: Optional[OAKDevice] = None
        self._camera_stream = None

    def start_streaming(self) -> None:
        sc = copy.deepcopy(self._stream)
        if "latency" not in sc:
            sc["latency"] = self._latency
        self._device = OAKDevice(
            oak_device_name=self._oak_device_name,
            oak_device_id=self._oak_device_id,
            fps=self._fps,
            latency=self._latency,
        )
        self._device.build_pipeline([sc])
        self._device.start()
        streams = self._device.get_streams()
        self._camera_stream = streams[0] if streams else None
        if self._camera_stream is not None:
            self._camera_stream.start_streaming()

    def stop_streaming(self) -> None:
        if self._camera_stream is not None:
            try:
                self._camera_stream.stop_streaming()
            except Exception:
                pass
            self._camera_stream = None
        if self._device is not None:
            try:
                self._device.stop()
            except Exception:
                pass
            self._device = None

    def get_frame(self):
        if self._camera_stream is None:
            return None
        return self._camera_stream.get_frame()


def select_camera(
    camera_type,
    device_id,
    oak_device_id=None,
    oak_device_name=None,
    iphone_device_name=None,
    iphone_udid=None,
):
    """Selects the appropriate camera class based on type and params."""
    if camera_type == "iphone":
        return IphoneCamera(
            iphone_device_name=iphone_device_name,
            iphone_udid=iphone_udid,
        )
    if camera_type == "uvc":
        return UVCCamera(camera_index=device_id, device_name="UVC Camera")
    if camera_type == "oak":
        return OAKCamera(
            oak_device_name=oak_device_name,
            oak_device_id=oak_device_id,
        )
    raise ValueError(f"Unknown camera type: {camera_type}")


def make_verify_camera(camera_type: str, kwargs: Optional[Dict[str, Any]]):
    """Build a camera-like object for marker verification."""
    if camera_type == "oak_device_stream":
        return OAKDeviceStreamVerifyCamera(**kwargs)
    return select_camera(
        camera_type,
        kwargs.get("device_id", 0),
        oak_device_id=kwargs.get("oak_device_id"),
        oak_device_name=kwargs.get("oak_device_name"),
        iphone_device_name=kwargs.get("iphone_device_name"),
        iphone_udid=kwargs.get("iphone_udid"),
    )


# Share camera-role aliases with the server so decoded events are folded
# into the same source slots that the UI displays.
_ROLE_TO_SOURCE_NAMES = ROLE_SOURCE_NAMES

# Human-readable label for each role (for logs and the preview)
ROLE_LABELS = {
    "left_tracking": "Left tracking (iPhone)",
    "left_umi": "Left UMI (OAK)",
    "left_trigger": "Left trigger",
    "right_tracking": "Right tracking (iPhone)",
    "right_umi": "Right UMI (OAK)",
    "right_trigger": "Right trigger",
    "head": "Head tracking (iPhone)",
    "head_umi": "Head UMI (OAK)",
}


def _unresolved(role, reason):
    """Report a role that will not be verified, then skip it.

    Reported rather than skipped silently, so renaming a source cannot turn the
    verification into a no-op that still reports success.
    """
    print(
        f"! {ROLE_LABELS.get(role, role)}: not verified — {reason}.",
        file=sys.stderr,
    )
    return None, None


def resolve_camera_for_role(config_path, role):
    """
    Load config and return (camera_type, kwargs) for the given role, or (None, None) if skip.
    role: left_tracking | left_umi | left_trigger | right_tracking | right_umi |
          right_trigger | head | head_umi
    """
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    names = _ROLE_TO_SOURCE_NAMES.get(role)
    if not names:
        return _unresolved(role, f"unknown role {role!r}")

    if role in HEAD_ROLES and not config.get("meta", {}).get("use_head_mount", False):
        return _unresolved(role, "meta.use_head_mount is not set")

    for src in config.get("video_sources", []) or []:
        # OAK-FFC-3P: logical camera names are under oak_device.streams[].name
        if src.get("type") == "oak_device":
            for stream in src.get("streams") or []:
                if stream.get("name") not in names:
                    continue
                stream_cfg: Dict[str, Any] = {
                    "name": stream["name"],
                    "socket": stream["socket"],
                    "resolution": list(stream["resolution"]),
                }
                if "latency" in stream:
                    stream_cfg["latency"] = stream["latency"]
                return "oak_device_stream", {
                    "oak_device_name": src.get("oak_device_name"),
                    "oak_device_id": src.get("oak_device_id"),
                    "fps": int(src.get("fps", 30)),
                    "latency": float(src.get("latency", 0.0)),
                    "stream": stream_cfg,
                }
            continue

        if src.get("name") not in names:
            continue
        stype = src.get("type")
        params = src.get("params", {})
        if stype == "oak_camera":
            return "oak", {
                "oak_device_name": params.get("oak_device_name"),
                "oak_device_id": params.get("oak_device_id"),
            }
        if stype == "uvc_camera":
            return "uvc", {"device_id": params.get("camera_index", 0)}
        if stype == "iphone_camera":
            return "iphone", {
                "iphone_device_name": params.get("iphone_device_name"),
                "iphone_udid": params.get("iphone_udid"),
            }
    return _unresolved(role, f"no video source named any of {list(names)}")


def payload_for_role(role):
    """Return the payload string for this role (UMI_VERIFY_LEFT / RIGHT / HEAD)."""
    if role in ("head", "head_umi"):
        return f"{PAYLOAD_PREFIX}HEAD"
    if role.startswith("left_"):
        return f"{PAYLOAD_PREFIX}LEFT"
    if role.startswith("right_"):
        return f"{PAYLOAD_PREFIX}RIGHT"
    return f"{PAYLOAD_PREFIX}{role.upper()}"


# ------------------------------------------------------------------ watch mode

# Prefix JSON events so they can share stdout with human-readable SDK logs.
# Teardown messages must not be parsed as verification events.
WATCH_PREFIX = "##station-ui-verify## "

# Limit empty-poll frequency so disconnected cameras cannot cause a busy loop.
WATCH_POLL_S = 0.01


@dataclass(frozen=True)
class DeviceGroup:
    """Every role that one physical device answers for.

    Grouping by device rather than by role is the point: an OAK-FFC carries
    left_umi and left_trigger behind one exclusive connection, and opening it
    once per role would open it twice in sequence.
    """

    key: tuple
    camera_type: str
    kwargs: dict
    roles: tuple
    streams: tuple = ()


def _group_key(camera_type, kwargs, role):
    """What makes two roles the same piece of hardware.

    A device that carries NO identifier is never the same device as another
    one: it falls back to a key only this role can hold. Merging on `None`
    reads as "these are one board" and is wrong in two different ways. Two
    unnamed OAK-FFCs would build a single pipeline on the first board and serve
    the second board's roles from the first board's sockets -- so the check
    oscillates forever, telling the operator two cameras are cross-wired when
    nothing is. Two unnamed iPhones would open only the first, and the second
    role would stay Waiting for as long as the operator kept aiming at it.
    """
    if camera_type in ("oak_device_stream", "oak"):
        identifier = kwargs.get("oak_device_name") or kwargs.get("oak_device_id")
    elif camera_type == "iphone":
        identifier = kwargs.get("iphone_udid") or kwargs.get("iphone_device_name")
    elif camera_type == "uvc":
        identifier = kwargs.get("device_id")
    else:
        identifier = None
    if identifier is None:
        return (camera_type, None, role)
    return (camera_type, identifier)


def plan_devices(config_path, roles):
    """Resolve every role, then fold the ones sharing a device into one group.

    A role the config does not declare contributes nothing, exactly as
    `resolve_camera_for_role` already decides -- it is reported there, on
    stderr, rather than silently dropped.
    """
    ordered = {}
    for role in roles:
        camera_type, kwargs = resolve_camera_for_role(config_path, role)
        if camera_type is None:
            continue
        key = _group_key(camera_type, kwargs, role)
        existing = ordered.get(key)
        if existing is None:
            ordered[key] = DeviceGroup(
                key=key,
                camera_type=camera_type,
                kwargs=kwargs,
                roles=(role,),
                streams=(kwargs["stream"],)
                if camera_type == "oak_device_stream"
                else (),
            )
            continue
        ordered[key] = DeviceGroup(
            key=key,
            camera_type=existing.camera_type,
            kwargs=existing.kwargs,
            roles=existing.roles + (role,),
            streams=(
                existing.streams + (kwargs["stream"],)
                if camera_type == "oak_device_stream"
                else existing.streams
            ),
        )
    return list(ordered.values())


class _MultiStreamWatcher:
    """One OAKDevice, every stream this board answers for, one pipeline.

    `OAKDevice.build_pipeline` takes a list of stream configs; the per-role
    path only ever passes one.
    """

    def __init__(self, group):
        self._group = group
        self._device = None
        self._streams = []

    def start(self):
        kwargs = self._group.kwargs
        configs = []
        for stream in self._group.streams:
            config = copy.deepcopy(stream)
            config.setdefault("latency", kwargs.get("latency", 0.0))
            configs.append(config)
        self._device = OAKDevice(
            oak_device_name=kwargs.get("oak_device_name"),
            oak_device_id=kwargs.get("oak_device_id"),
            fps=int(kwargs.get("fps", 30)),
            latency=float(kwargs.get("latency", 0.0)),
        )
        self._device.build_pipeline(configs)
        self._device.start()
        # build_pipeline preserves order, so stream i answers for role i.
        self._streams = list(self._device.get_streams())
        for stream in self._streams:
            stream.start_streaming()

    def frames(self):
        for role, stream in zip(self._group.roles, self._streams):
            data = stream.get_frame()
            if data is not None and data.rgb is not None:
                yield role, data.rgb

    def stop(self):
        for stream in self._streams:
            try:
                stream.stop_streaming()
            except Exception:
                pass
        self._streams = []
        if self._device is not None:
            try:
                self._device.stop()
            except Exception:
                pass
            self._device = None


class _SingleCameraWatcher:
    """An iPhone, a UVC camera, or a whole-device OAK: one role, one open."""

    def __init__(self, group):
        # Require exactly one role per camera watcher; otherwise a second role could
        # wait forever. Raise explicitly so the guard remains active under python -O.
        if len(group.roles) != 1:
            raise ValueError(
                f"{group.camera_type} device {group.key!r} was grouped with "
                f"{len(group.roles)} roles ({', '.join(group.roles)}); one "
                "camera can only answer for one role"
            )
        self._group = group
        self._camera = None

    def start(self):
        self._camera = make_verify_camera(self._group.camera_type, self._group.kwargs)
        self._camera.start_streaming()

    def frames(self):
        data = self._camera.get_frame()
        if data is not None and data.rgb is not None:
            yield self._group.roles[0], data.rgb

    def stop(self):
        if self._camera is not None:
            try:
                self._camera.stop_streaming()
            except Exception:
                pass
            self._camera = None


def make_watcher(group):
    if group.camera_type == "oak_device_stream":
        return _MultiStreamWatcher(group)
    return _SingleCameraWatcher(group)


def _emit(out, **event):
    print(WATCH_PREFIX + json.dumps(event), file=out, flush=True)


# Optional native previews show the cameras for the current side. Treat
# display failures as nonfatal: verification depends on the event stream,
# not on a graphical session.

PREVIEW_SIDES = ("left", "right", "head")
# Preview tile height in screen pixels.
PREVIEW_TILE_H = 960
# Bound the preview window to the monitor while composing a full-resolution
# canvas for resizing. Preserve the canvas aspect ratio.
PREVIEW_MAX_W = 1920
PREVIEW_MAX_H = 1080


def _side_of_role(role):
    if role in ("head", "head_umi"):
        return "head"
    return role.split("_", 1)[0]


def _current_preview_side(roles, verified):
    """The first side with a camera still to verify: the one to preview now.

    `verified` is the latched set of roles that have decoded their own code,
    not a snapshot of what each camera last saw. The cameras stay open after
    their side clears, so the right-hand rig decodes `UMI_VERIFY_HEAD` as the
    operator carries the code past it on the way to the head mount. Judged on
    what each camera last saw, the right side would un-verify itself and this
    window would snap back to RIGHT, so the head preview would never open.

    `station_ui.calibration.fold_events` latches for the same reason; this is
    the same rule on the child's side of the pipe, so the window and the panel
    always agree on which side the operator is on.
    """
    for side in PREVIEW_SIDES:
        side_roles = [r for r in roles if _side_of_role(r) == side]
        if side_roles and any(r not in verified for r in side_roles):
            return side
    return None


def _draw_text(img, text, org, color=(255, 255, 255)):
    # Drawn twice: a dark stroke under the fill so the label stays legible over
    # a bright or busy frame.
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA
    )
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _label_tile(frame, label, matched):
    """One camera's frame, normalised to PREVIEW_TILE_H, with a role label and a
    verified/aim-here border."""
    img = np.ascontiguousarray(frame)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    h, w = img.shape[:2]
    scale = PREVIEW_TILE_H / float(h) if h else 1.0
    img = cv2.resize(img, (max(1, int(round(w * scale))), PREVIEW_TILE_H))
    h, w = img.shape[:2]
    # BGR: green when this camera has verified, amber while it still needs aiming.
    color = (0, 185, 0) if matched else (0, 180, 255)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 8)
    _draw_text(img, label, (14, 34))
    _draw_text(img, "VERIFIED" if matched else "AIM HERE", (14, h - 18), color)
    return img


def _waiting_tile(label):
    """A camera that has not returned a frame yet: the cameras are still
    starting. A placeholder so the window is never blank while they connect."""
    w = int(round(PREVIEW_TILE_H * 4 / 3))
    tile = np.full((PREVIEW_TILE_H, w, 3), 32, np.uint8)
    color = (0, 180, 255)  # amber, the same as a camera still to be aimed
    cv2.rectangle(tile, (0, 0), (w - 1, PREVIEW_TILE_H - 1), color, 8)
    _draw_text(tile, label, (14, 34))
    _draw_text(tile, "STARTING...", (14, PREVIEW_TILE_H - 18), color)
    return tile


def _compose_preview(latest_frame, verified, roles, label_for):
    """The image to show for the side being aimed now, or None when no side is
    outstanding.

    One tile per camera on that side: its live frame once it arrives, and a
    'starting' placeholder until then, so the window is never blank while the
    cameras connect. Pure (numpy plus cv2 drawing only, no window), so it is
    unit-testable without a display.

    There is no marker tile: the operator aims the marker the browser serves
    (`/calibration/check/qr/{side}.svg`) and this monitor is behind them, so
    the whole width goes to the camera feeds, the one thing the page cannot
    show.
    """
    side = _current_preview_side(roles, verified)
    if side is None:
        return None
    tiles = []
    for role in roles:
        if _side_of_role(role) != side:
            continue
        frame = latest_frame.get(role)
        if frame is not None:
            tiles.append(_label_tile(frame, label_for(role), role in verified))
        else:
            tiles.append(_waiting_tile(label_for(role)))
    if not tiles:
        return None
    return _grid(tiles)


# Use at most two cameras per row so each view remains large enough to aim.
PREVIEW_MAX_COLS = 2


def _grid(tiles):
    """The tiles laid out at most `PREVIEW_MAX_COLS` per row, as one image.

    Every tile shares `PREVIEW_TILE_H`, so a row hstacks cleanly; a short last
    row is padded to the widest row so the rows vstack. Pure numpy.
    """
    if len(tiles) == 1:
        return tiles[0]
    cols = min(PREVIEW_MAX_COLS, len(tiles))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    width = max(row.shape[1] for row in rows)
    padded = [
        row
        if row.shape[1] == width
        else np.hstack(
            [row, np.full((row.shape[0], width - row.shape[1], 3), 32, np.uint8)]
        )
        for row in rows
    ]
    return np.vstack(padded)


def _window_size(canvas_w, canvas_h):
    """The window size for a canvas: its own, shrunk to fit `PREVIEW_MAX_*`.

    Aspect-preserving, and never enlarges a small canvas: a one-camera side
    blown up to fill the monitor is just a blurry frame.
    """
    if canvas_w <= 0 or canvas_h <= 0:
        return None
    scale = min(PREVIEW_MAX_W / float(canvas_w), PREVIEW_MAX_H / float(canvas_h), 1.0)
    return max(1, int(round(canvas_w * scale))), max(1, int(round(canvas_h * scale)))


class _PreviewFile:
    """Opt-in browser preview from the check's already-open cameras.

    Overwrite one JPEG atomically, at most five times a second. This never
    creates a display or opens a second camera connection.
    """

    def __init__(self, path, label_for):
        self._path = str(path)
        self._label_for = label_for
        self._roles = []
        self._last_write = float("-inf")
        self._ok = True

    def begin(self, roles):
        self._roles = list(roles)

    def render(self, latest_frame, verified):
        if not self._ok:
            return
        try:
            self._render(latest_frame, verified)
        except Exception as exc:  # noqa: BLE001 - preview must not stop the check
            self._ok = False
            print(f"[preview] unavailable: {exc}", file=sys.stderr, flush=True)

    def _render(self, latest_frame, verified):
        now = time.monotonic()
        if now - self._last_write < 0.2:
            return
        self._last_write = now
        canvas = _compose_preview(latest_frame, verified, self._roles, self._label_for)
        if canvas is None:
            return
        # Match the browser's maximum display width without saving huge frames.
        if canvas.shape[1] > 1280:
            height = round(canvas.shape[0] * 1280 / canvas.shape[1])
            canvas = cv2.resize(canvas, (1280, height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        partial = self._path + ".part"
        with open(partial, "wb") as handle:
            handle.write(encoded.tobytes())
        os.replace(partial, self._path)

    def close(self):
        # The last frame stays readable until the Station Console retires this run.
        pass


class _PreviewWindow:
    """A native OpenCV window over `_compose_preview`, one per side.

    The window is titled for the side being aimed (LEFT, then RIGHT, then HEAD)
    and is destroyed and recreated when the side changes, so the operator sees
    it close and reopen as each side clears, as a sign of progress. Best-effort
    throughout: a failed window call disables the preview for the rest of the
    run rather than raising, because the check's real job is the event stream
    and a preview that cannot draw must never take it down.
    """

    def __init__(self, label_for=None):
        self._label_for = label_for or (lambda role: role)
        self._roles = []
        self._ok = True
        self._side = None  # the side whose window is open now, or None
        self._window = None  # its window name, or None when none is open
        self._sized = None  # the canvas size the window was last fitted to

    def begin(self, roles):
        # The window opens lazily, when the first side becomes current, so a rig
        # with nothing to check never flashes an empty frame.
        self._roles = list(roles)

    def _switch_to(self, side):
        # Close the outgoing side's window before opening the next preview.
        if self._window is not None:
            try:
                cv2.destroyWindow(self._window)
            except Exception:
                pass
            self._window = None
        self._side = side
        self._sized = None
        if side is None:
            return
        try:
            name = f"UMI camera check — {side.upper()}"
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            self._window = name
        except Exception:
            self._ok = False

    def render(self, latest_frame, verified):
        if not self._ok or not self._roles:
            return
        try:
            side = _current_preview_side(self._roles, verified)
            if side != self._side:
                self._switch_to(side)
            if self._window is not None:
                canvas = _compose_preview(
                    latest_frame, verified, self._roles, self._label_for
                )
                if canvas is not None:
                    # Resize a WINDOW_NORMAL preview only when canvas dimensions change.
                    # Per-frame resizing would override the operator's manual window
                    # size.
                    shape = canvas.shape[:2]
                    if shape != self._sized:
                        self._sized = shape
                        fitted = _window_size(canvas.shape[1], canvas.shape[0])
                        if fitted is not None:
                            cv2.resizeWindow(self._window, fitted[0], fitted[1])
                    cv2.imshow(self._window, canvas)
            # Pump the GUI event loop even when there is nothing new to show, or
            # the window stops repainting and reads as frozen.
            cv2.waitKey(1)
        except Exception:
            self._ok = False

    def close(self):
        if self._window is not None:
            try:
                cv2.destroyWindow(self._window)
            except Exception:
                pass
            self._window = None


def watch(
    groups,
    out,
    *,
    make_watcher=make_watcher,
    decode=None,
    should_stop=None,
    preview=None,
):
    """Hold every camera open and report what each one decodes, until stopped.

    This process never learns which marker is on the operator's screen. It reports
    "left_umi decoded UMI_VERIFY_RIGHT" and lets the server call that a
    cross-wired camera, a diagnosis a per-role timeout cannot make.

    One event per change, not per decoded frame. Re-aiming a camera that was
    pointed at the wrong code still corrects the earlier reading (the only
    retry this check has), but a camera parked on one code reports it once.
    Emitting per frame would put ~30 lines a second into the job's line buffer,
    which holds 20,000 lines and trims from the front, so a long-running check
    would evict its own `open_failed` lines.
    """
    # Try ArUco on the full frame and enlarged center. Keep decoding
    # injectable for hardware-independent tests.
    decode = decode_marker if decode is None else decode
    should_stop = (lambda: False) if should_stop is None else should_stop

    # Keep roles with their plan group so watchers need only implement
    # start, frames and stop.
    live = []
    for group in groups:
        watcher = None
        try:
            # Construction is inside the try too: a Record3D device that is
            # already claimed fails when the watcher is built, not on start().
            watcher = make_watcher(group)
            watcher.start()
        except Exception as exc:  # noqa: BLE001 - the reason is the whole point
            # A partial start can leave the camera claimed; stop the watcher before
            # propagating the failure.
            if watcher is not None:
                try:
                    watcher.stop()
                except Exception:
                    pass
            # One dead device must not take the rest of the rig's check with it.
            for role in group.roles:
                _emit(
                    out,
                    event="open_failed",
                    role=role,
                    error=f"{type(exc).__name__}: {exc}",
                )
            continue
        live.append((group, watcher))

    ready_roles = [role for group, _ in live for role in group.roles]
    _emit(out, event="ready", roles=ready_roles)
    if preview is not None:
        preview.begin(ready_roles)

    # Cache the last decoded payload only to suppress duplicate events.
    last_seen: Dict[str, str] = {}
    # Latch each verified role for preview progression; later frames of another
    # side must not undo a completed match.
    verified: set = set()
    latest_frame: Dict[str, Any] = {}
    try:
        while not should_stop():
            for _group, watcher in live:
                for role, frame in watcher.frames():
                    payload = decode(frame)
                    if payload and last_seen.get(role) != payload:
                        last_seen[role] = payload
                        _emit(out, event="saw", role=role, payload=payload)
                    if payload and payload == payload_for_role(role):
                        verified.add(role)
                    if preview is not None:
                        latest_frame[role] = frame
            if preview is not None:
                preview.render(latest_frame, verified)
            # Throttle cameras that return None immediately; unlike OAK queue reads,
            # those calls provide no blocking delay.
            time.sleep(WATCH_POLL_S)
    finally:
        for _group, watcher in live:
            watcher.stop()
        if preview is not None:
            preview.close()
    return 0


def install_stop_on_sigterm():
    """Turn SIGTERM into a loop exit, and return the predicate `watch` polls.

    Stop on the page is `jobs.cancel`, which sends SIGTERM first. Python's
    default SIGTERM handling ends the interpreter without unwinding, so the
    `finally` in `watch` that stops every camera would not run, and anything
    that outlives the kill (`uv run` sits between the job and this script)
    would keep holding an exclusive DepthAI connection. With this handler the
    cleanup block releases them.

    Like `umi.common.utility.signal_handling.install_stop_on_sigterm()`, except
    that one raises KeyboardInterrupt and this one sets a flag the `watch` loop
    polls.
    """
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stopping.set())
    return stopping.is_set


def main():
    parser = argparse.ArgumentParser(
        description="Verify device assignment: camera for role must see expected marker."
    )
    parser.add_argument("--config", required=True, help="Path to recording config YAML")
    parser.add_argument(
        "--watch",
        action="store_true",
        required=True,
        help="Hold every declared camera open and report what each decodes. "
        "Draws no window: the markers are shown in the station web UI.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=list(ROLE_LABELS),
        help="With --watch, verify only these roles; other roles were already identified.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="With --watch, open a native OpenCV window on the station monitor "
        "showing the cameras for the side being aimed now, so the operator can "
        "line the marker up in-frame. Best-effort: a no-op where there is no display.",
    )
    parser.add_argument(
        "--preview-file",
        help="With --watch, write the live preview to this JPEG for an embedded "
        "browser view instead of opening a window. Used by the station GUI.",
    )
    args = parser.parse_args()
    if args.preview and args.preview_file:
        parser.error("choose either --preview or --preview-file")

    groups = plan_devices(args.config, args.roles or list(ROLE_LABELS))
    if not groups:
        print("No cameras to check in this config.", file=sys.stderr)
        sys.exit(1)

    def label_for(role):
        return ROLE_LABELS.get(role, role)

    if args.preview_file:
        preview = _PreviewFile(args.preview_file, label_for)
    elif args.preview:
        preview = _PreviewWindow(label_for=label_for)
    else:
        preview = None
    sys.exit(
        watch(
            groups,
            sys.stdout,
            should_stop=install_stop_on_sigterm(),
            preview=preview,
        )
    )


if __name__ == "__main__":
    main()
