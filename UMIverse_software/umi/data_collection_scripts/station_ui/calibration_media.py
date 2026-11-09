# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser-only presentation of the calibration workers' existing samples."""

from __future__ import annotations

import math
from collections import deque


def tracker_chart(events):
    """Bounded movement traces, with gaps where a tracker has lost tracking.

    Both arms stay visible: moving the wrong tracker is exactly the fault this
    check is intended to catch. No hardware or plotting backend is opened here.
    """
    samples = deque(maxlen=150)
    move_m, still_m = 0.05, 0.01
    for event in events:
        if event.get("event") == "ready":
            for key in ("move_m", "still_m"):
                value = event.get(key)
                if _number(value) and value > 0:
                    if key == "move_m":
                        move_m = value
                    else:
                        still_m = value
        elif event.get("event") == "sample" and _number(event.get("t")):
            samples.append(event)
    if not samples:
        return None
    start = samples[0]["t"]
    end = max(start + 1, samples[-1]["t"])
    arms = sorted({arm for sample in samples for arm in (sample.get("arms") or {})})
    maximum = max(
        [move_m * 1.5, still_m * 1.5]
        + [
            entry["disp"]
            for sample in samples
            for entry in (sample.get("arms") or {}).values()
            if isinstance(entry, dict) and _number(entry.get("disp"))
        ]
    )

    def y(value):
        return round(176 - max(0, value) / maximum * 140, 1)

    traces = []
    for arm in arms:
        paths, points = [], []
        for sample in samples:
            entry = (sample.get("arms") or {}).get(arm) or {}
            value = entry.get("disp") if isinstance(entry, dict) else None
            if not _number(value) or not entry.get("tracked"):
                if points:
                    paths.append(" ".join(points))
                    points = []
                continue
            x = 52 + (sample["t"] - start) / (end - start) * 560
            points.append(f"{x:.1f},{y(value):.1f}")
        if points:
            paths.append(" ".join(points))
        traces.append({"arm": arm, "paths": paths})
    return {
        "traces": traces,
        "move_y": y(move_m),
        "still_y": y(still_m),
        "move_mm": round(move_m * 1000),
        "still_mm": round(still_m * 1000),
        "max_mm": round(maximum * 1000),
        "start": round(start, 1),
        "end": round(end, 1),
    }


def _number(value):
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
