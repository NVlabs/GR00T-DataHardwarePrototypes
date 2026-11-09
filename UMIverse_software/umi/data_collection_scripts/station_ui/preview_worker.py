# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream one camera into a JPEG file, for the configuration page's live preview.

Run as `python -m station_ui.preview_worker --kind oak --device <id> --out <file>`.
A child of `station_ui.jobs` for the reason the scan is one: every camera open
in this repo is untimed and an OAK open is exclusive, so in-process it would pin
a request handler; as a capped child it is a pid the console can cancel, with a
wall-clock cap for the pop-up nobody closed.

It does one thing: write the newest picture over `--out` (atomically, through a
sibling temp file). `station_ui.routes.preview` streams that file to the
browser. An OAK board shows every sensor it has -- not the phantom CAM_C a V4
gripper board lists -- side by side and labelled with its socket and sensor
module. A USB camera shows its one picture, labelled with its `/dev/video`
node: a gripper's trigger camera is plugged into the computer rather than the
gripper, so its picture is what says which gripper it is under.

Runs until SIGTERM; the `finally` closes the camera so the next user of it --
a scan, the camera check, a recording -- can open it.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

from station_ui.core.device_scan import (
    DEFAULT_STREAM_RESOLUTION,
    SENSOR_STREAM_RESOLUTION,
)

__all__ = ["KINDS", "compose", "main", "run", "write_jpeg"]

KINDS = ("oak", "iphone", "uvc")
PREFIX = "[preview]"

# Limit preview rate to approximately 10 Hz to reduce shared USB load.
FRAME_INTERVAL_S = 0.1
OAK_FPS = 15
# The mode `bimanual_v4.yaml` records a trigger camera in, so a USB camera that
# previews is one the recording can open the same way.
UVC_RESOLUTION = (1280, 720)
UVC_FPS = 30
# Tile heights per kind. A phone is portrait, so it gets more height to be the
# same size on screen as one landscape OAK tile.
TILE_H = {"oak": 360, "iphone": 640, "uvc": 360}
MAX_WIDTH = 1600
JPEG_QUALITY = 80
_GAP = 4

STARTING = "starting..."
# Bound first-frame wait for sensors that enumerate but never stream.
NO_PICTURE_AFTER_S = 3.0
NO_PICTURE = "no picture from this sensor"
# Stereo mono sensors may advertise color support; use grayscale rather
# than a color pipeline that would produce a magenta image.
MONO_SENSORS = {"OV9282", "OV9281", "OV7251"}


# ------------------------------------------------------------------ picture


def _draw_text(img, text, org):
    # A dark stroke under the fill, so the label reads over a bright scene --
    # the same treatment as the camera check's preview.
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA
    )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )


def _tile(label, frame, height, note=STARTING):
    if frame is None:
        # No picture (yet). A placeholder keeps the layout from jumping when
        # the first frame lands, and says which kind of empty it is.
        tile = np.full((height, round(height * 16 / 10), 3), 32, np.uint8)
        _draw_text(tile, note, (14, height - 18))
    else:
        img = np.ascontiguousarray(frame)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        tile = cv2.resize(
            img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA
        )
    if label:
        _draw_text(tile, label, (14, 34))
    return tile


def compose(tiles, height=TILE_H["oak"]):
    """`[(label, BGR or grey frame or None[, note])]` as one image: equal
    heights, side by side. `note` is what a tile with no frame says.

    None when there is nothing to show at all.
    """
    images = []
    for label, frame, *note in tiles:
        images.append(_tile(label, frame, height, *note))
    if not images:
        return None
    gap = np.zeros((height, _GAP, 3), np.uint8)
    parts = []
    for image in images:
        if parts:
            parts.append(gap)
        parts.append(image)
    canvas = np.hstack(parts)
    if canvas.shape[1] > MAX_WIDTH:
        scaled_h = max(1, round(canvas.shape[0] * MAX_WIDTH / canvas.shape[1]))
        canvas = cv2.resize(canvas, (MAX_WIDTH, scaled_h), interpolation=cv2.INTER_AREA)
    return canvas


def write_jpeg(image, path) -> bool:
    """Replace `path` with `image`, never leaving a half-written file to read."""
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return False
    partial = f"{path}.part"
    with open(partial, "wb") as handle:
        handle.write(buffer.tobytes())
    os.replace(partial, path)
    return True


# ------------------------------------------------------------------ cameras


def _phantom(socket, sensors) -> bool:
    """Whether *socket* is the V4 gripper baseboard's phantom CAM_C.

    That board wires CAM_B and CAM_C to one I2C bus, so its one module answers
    on both: no CAM_A, and CAM_B's sensor again on CAM_C. A node on that CAM_C
    never sends a picture, and with it in the pipeline CAM_B's picture blows out
    to white even as CAM_B's auto-exposure falls to its minimum: the two sockets
    drive one sensor. Any other CAM_C -- a head OAK-D's mono camera -- is real.
    """
    return (
        socket == "CAM_C"
        and "CAM_A" not in sensors
        and sensors["CAM_C"] == sensors.get("CAM_B")
    )


class _OakSource:
    """Every sensor on one board, through the recorder's own `OAKDevice`.

    One reader thread per sensor. `get_frame` waits up to a second on a sensor
    with nothing to send, and read in turn, one silent sensor would slow every
    tile to a picture a second.
    """

    def __init__(self, device_id):
        from umi.common.hardware_sdk.camera.oak_device import OAKDevice

        self._device = OAKDevice(oak_device_id=device_id, fps=OAK_FPS)
        try:
            sensors = {
                feature.socket.name: feature.sensorName
                for feature in self._device.device.getConnectedCameraFeatures()
            }
            if not sensors:
                raise RuntimeError("the board reports no camera")
            sockets = sorted(s for s in sensors if not _phantom(s, sensors))
            self._device.build_pipeline(
                [
                    {
                        "name": socket,
                        "socket": socket,
                        "resolution": SENSOR_STREAM_RESOLUTION.get(
                            sensors[socket], DEFAULT_STREAM_RESOLUTION
                        ),
                    }
                    for socket in sockets
                ]
            )
            self._device.start()
        except Exception:
            self._device.stop()
            raise
        self._labels = [f"{socket} {sensors[socket]}" for socket in sockets]
        self._mono = [sensors[socket] in MONO_SENSORS for socket in sockets]
        streams = self._device.get_streams()
        self._latest = [None] * len(streams)
        self._opened = time.monotonic()
        self._stopping = threading.Event()
        self._readers = [
            threading.Thread(target=self._read, args=(index, stream), daemon=True)
            for index, stream in enumerate(streams)
        ]
        for reader in self._readers:
            reader.start()

    def _read(self, index, stream):
        while not self._stopping.is_set():
            data = stream.get_frame()
            if data is None or data.rgb is None:
                continue
            frame = data.rgb
            if self._mono[index]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._latest[index] = frame

    def tiles(self):
        silent = time.monotonic() - self._opened > NO_PICTURE_AFTER_S
        return [
            (label, frame)
            if frame is not None or not silent
            else (label, None, NO_PICTURE)
            for label, frame in zip(self._labels, self._latest)
        ]

    def close(self):
        self._stopping.set()
        for reader in self._readers:
            reader.join(timeout=2.0)
        self._device.stop()


class _IphoneSource:
    """The phone's colour camera, through Record3D. Addressed by UDID, as the
    scan's probe does, so an unregistered phone can be previewed too."""

    def __init__(self, udid):
        from umi.common.hardware_sdk.camera.iphone_camera import IphoneCamera

        self._camera = IphoneCamera(iphone_udid=udid)
        self._camera.start_streaming()
        self._latest = None

    def tiles(self):
        data = self._camera.get_frame()
        if data is not None and data.rgb is not None:
            self._latest = data.rgb
        return [("iPhone", self._latest)]

    def close(self):
        self._camera.stop_streaming()


class _UvcSource:
    """A USB (UVC) camera by its V4L2 index, through the recorder's own
    `UVCCamera`.

    Read on a thread of its own, like an OAK sensor: a read waits for the next
    frame -- for seconds, on a camera that has stalled -- and that must not
    stall the loop that writes the picture.
    """

    def __init__(self, index):
        from umi.common.hardware_sdk.camera.uvc_camera import UVCCamera

        self._label = f"/dev/video{index}"
        self._camera = UVCCamera(
            camera_index=int(index),
            device_name=self._label,
            camera_resolution=UVC_RESOLUTION,
            fps=UVC_FPS,
        )
        self._camera.start_streaming()
        self._latest = None
        self._opened = time.monotonic()
        self._stopping = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        try:
            while not self._stopping.is_set():
                data = self._camera.get_frame()
                if data is None or data.rgb is None:
                    # A failed read returns at once: without the pause, an
                    # unplugged camera would spin a core.
                    time.sleep(FRAME_INTERVAL_S)
                    continue
                self._latest = data.rgb
        finally:
            # Release the camera on its reader thread after the last read. Closing it
            # concurrently with a blocked SDK read can crash the process.
            self._camera.stop_streaming()

    def tiles(self):
        silent = time.monotonic() - self._opened > NO_PICTURE_AFTER_S
        if self._latest is None and silent:
            return [(self._label, None, NO_PICTURE)]
        return [(self._label, self._latest)]

    def close(self):
        # A reader still stuck after this dies with the worker, whose exit
        # closes the device.
        self._stopping.set()
        self._reader.join(timeout=2.0)


def _open(kind, device):
    """The source for one device. Looked up at call time so a test can replace it."""
    if kind == "oak":
        return _OakSource(device)
    if kind == "uvc":
        return _UvcSource(device)
    return _IphoneSource(device)


# --------------------------------------------------------------------- loop


def run(source, out, height, should_stop) -> None:
    """Write a picture every `FRAME_INTERVAL_S` until `should_stop()` says so."""
    while not should_stop():
        started = time.monotonic()
        image = compose(source.tiles(), height)
        if image is not None:
            write_jpeg(image, out)
        time.sleep(max(0.0, FRAME_INTERVAL_S - (time.monotonic() - started)))


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _arguments(sys.argv[1:] if argv is None else argv)
    stopping = []

    def _stop(_signum, _frame):
        stopping.append(True)

    # A flag rather than an exception: the loop may be inside an SDK call, and
    # the camera has to be closed by the `finally` below, not abandoned mid-read.
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print(f"{PREFIX} opening {args.kind} {args.device}...", flush=True)
    try:
        source = _open(args.kind, args.device)
    except Exception as exc:  # noqa: BLE001 - the reason is the whole message
        print(f"{PREFIX} error: could not open {args.device}: {exc}", flush=True)
        return 1
    print(f"{PREFIX} streaming {args.device}", flush=True)
    try:
        run(source, args.out, TILE_H[args.kind], lambda: bool(stopping))
    except Exception as exc:  # noqa: BLE001
        print(f"{PREFIX} error: the camera stopped: {exc}", flush=True)
        return 1
    finally:
        try:
            source.close()
        except Exception:  # noqa: BLE001 - closing is best-effort on the way out
            pass
        try:
            os.remove(args.out)
        except OSError:
            pass
    print(f"{PREFIX} stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
