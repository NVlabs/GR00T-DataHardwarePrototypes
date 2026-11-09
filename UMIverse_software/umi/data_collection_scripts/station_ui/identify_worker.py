# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe fresh frames from one camera; SDKs stay in this cancellable child."""

import argparse
import json
import os
import signal
import sys
import time

PREFIX = "[identify] "


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("oak", "iphone", "uvc"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    # Import native libraries only after installing the shutdown handlers.
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if sys.platform == "linux" and args.parent_pid is not None:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if (
            libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0
            or os.getppid() != args.parent_pid
        ):
            return 1

    def emit(**event):
        print(PREFIX + json.dumps(event), flush=True)

    source = None
    try:
        import cv2

        from station_ui import markers
        from station_ui.preview_worker import _open, compose, write_jpeg

        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        )
        source = _open(args.kind, args.device)
        emit(ready=True)
        previous = {}
        seq = 0
        deadline = time.monotonic() + 300
        while not stopping and time.monotonic() < deadline:
            tiles = source.tiles()
            found, fresh = set(), False
            for label, frame, *_ in tiles:
                if frame is None or previous.get(label) is frame:
                    continue
                previous[label] = frame  # retain the reference: no cached-frame votes
                fresh = True
                gray = (
                    frame
                    if frame.ndim == 2
                    else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                )
                _, ids, _ = detector.detectMarkers(gray)
                if ids is not None:
                    found.update(int(value) for value in ids.ravel())
            if fresh:
                seq += 1
                image = compose(tiles, height=240)
                if image is not None:
                    write_jpeg(image, args.out)
                emit(
                    seq=seq,
                    at=time.monotonic(),
                    sides=[
                        side
                        for side, marker in markers.MARKER_IDS.items()
                        if marker in found
                    ],
                )
            time.sleep(0.1)
    except Exception as exc:
        emit(error=str(exc))
        return 1
    finally:
        if source is not None:
            source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
