# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove each arm's gripper drives the Vive tracker the config names.

A rig whose `left_tracking_pose` is bound to the right arm's puck records a
whole session of mislabelled pose and passes every data check: pose_movement,
pose_jumps and consecutive_movement all see plausible motion on both arms, and
nothing in the pipeline can tell the two apart afterwards. It has to be caught
at the bench, which is what this check does.

The operator is walked through one arm at a time. An arm is confirmed when its
tracker moves while every other tracker stays still, with both conditions held
for a second: a single frame of clean separation happens by accident, and a
knock against the other gripper produces a transient rather than a second of
mutual exclusivity.

Two things are kept separate:

* `amplitude` and `classify` are pure. They are the whole verdict, and they are
  testable with synthetic numbers: no ZMQ, no matplotlib, no puck.
* the poll loop below them is the shell: it talks to the bridge, prints events
  and draws the plot, and makes no decisions of its own.

Run by the Station Console as
`verify_vive_assignment.py --config <generated> --watch --plot`, and by hand at
the bench with the same arguments.
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import deque

import yaml

from umi.common.hardware_sdk.vive.client import ViveTrackerClient
from umi.common.hardware_sdk.vive.tracker import ViveTracker
from umi.common.utility.tracking import tracking_device_side

# Use a distinct JSON-event prefix so camera and Vive checks cannot parse
# each other's output.
VIVE_PREFIX = "##station-ui-vive## "

# Motion amplitudes use meters. Leave a gap between still and moving
# thresholds so tracker drift is not forced into either verdict.
MOVE_M = 0.05
STILL_M = 0.01

# How long both halves must hold together before a verdict is reached.
HOLD_S = 1.0

# The rolling window `amplitude` measures over.
WINDOW_S = 1.0

# Poll more slowly than recording requires to limit load on the shared
# ZMQ REP bridge while retaining responsive motion feedback.
POLL_HZ = 20.0

# Throttle browser updates separately from bridge polling to bound SSE
# traffic and preserve useful history in the job log.
EMIT_HZ = 5.0


def amplitude(samples) -> float:
    """Span of motion over a window of ``(t, (x, y, z))`` samples, in metres.

    The span (max - min per axis, combined), not the net displacement from the
    first sample to the last. An operator who moves the gripper around and puts
    it back down has moved it, and net displacement scores that zero, which
    would fail a check the operator performed correctly. The span is also
    immune to slow lighthouse drift, because drift moves both ends of the
    window together.
    """
    points = [point for _t, point in samples]
    if not points:
        return 0.0
    spans = [
        max(p[axis] for p in points) - min(p[axis] for p in points) for axis in range(3)
    ]
    return math.sqrt(sum(span * span for span in spans))


def classify(amplitudes, prompted: str, held_s: float, thresholds):
    """``"confirmed"``, ``"cross"``, or ``None``: the whole verdict.

    ``amplitudes`` maps arm -> metres of motion, or ``None`` for an arm whose
    puck the server is not tracking (which can never confirm: there is no
    position to judge).

    Both outcomes need the same shape of evidence (one arm at or above MOVE_M
    while every other is at or below STILL_M, held for HOLD_S) and differ only
    in which arm it was. Literally the same bar, boundary included: the two
    comparisons are both `>=`, because an arm that would confirm at exactly
    MOVE_M must be able to cross at exactly MOVE_M as well.

    The symmetry is deliberate: it stops a bench knock against the other
    gripper from reading as swapped keys, because a knock is a transient and
    does not hold, and it stops a two-handed operator from confirming anything
    at all.
    """
    move_m, still_m, hold_s = thresholds
    if held_s < hold_s:
        return None

    others = {arm: value for arm, value in amplitudes.items() if arm != prompted}
    # An untracked puck has unknown amplitude, not zero motion; it cannot confirm
    # stillness.
    if any(value is None or value > still_m for value in others.values()):
        # Use the same inclusive movement threshold for confirmation and swap detection.
        moving = [
            arm
            for arm, value in others.items()
            if value is not None and value >= move_m
        ]
        # Any ambiguous or untracked arm prevents proving that exactly one other arm
        # moved.
        unsettled = [
            arm
            for arm, value in others.items()
            if arm not in moving and (value is None or value > still_m)
        ]
        prompted_value = amplitudes.get(prompted)
        # A swap requires the prompted arm to be still, exactly one other arm to
        # move, and no ambiguous or untracked arms.
        if (
            len(moving) == 1
            and not unsettled
            and prompted_value is not None
            and prompted_value <= still_m
        ):
            return "cross"
        return None

    prompted_value = amplitudes.get(prompted)
    if prompted_value is not None and prompted_value >= move_m:
        return "confirmed"
    return None


def _emit(out, **event):
    print(VIVE_PREFIX + json.dumps(event), file=out, flush=True)


def plan_sources(config_path):
    """The Vive trackers this rig records, one entry per arm.

    Reads the generated config, never a template: the templates carry a
    placeholder bridge address and keys, and a check pointed at those proves
    nothing about this station.
    """
    with open(os.path.expanduser(config_path)) as handle:
        cfg = yaml.safe_load(handle) or {}
    sources = []
    for entry in cfg.get("numeric_sources") or []:
        if entry.get("type") != "vive_tracker":
            continue
        arm = tracking_device_side(str(entry.get("name") or ""))
        if arm is None:
            continue
        params = entry.get("params") or {}
        sources.append(
            {
                "name": entry.get("name"),
                "arm": arm,
                "endpoint": str(params.get("endpoint") or ""),
                "tracker_key": str(params.get("tracker_key") or ""),
            }
        )
    return sources


def install_stop_on_sigterm():
    """A `should_stop()` that goes true on SIGTERM.

    `jobs.start` runs this child with `start_new_session=True` and `jobs.cancel`
    signals the whole group, so without this the process dies without
    unwinding: the poll thread and the REQ socket are released only because the
    process image goes away, and no final event line is printed.

    A local copy rather than an import, as in `verify_device_assignment.py`.
    Importing it from there would drag `cv2` into a child that opens no camera,
    and `umi.common.utility.signal_handling`'s version raises KeyboardInterrupt
    where this loop wants a flag to poll.
    """
    stopped = {"now": False}

    def _stop(_signum, _frame):
        stopped["now"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return lambda: stopped["now"]


def open_plot(arms):
    """A live figure on the station monitor, or None. Never raises.

    The guard cannot be a try/except. matplotlib is a declared dependency and
    never fails headless: with DISPLAY unset it silently resolves to Agg, so
    the import succeeds and plt.pause() runs twenty times a second drawing into
    nothing. Both checks below are needed, and the second is not redundant: the
    only interactive backend this environment can reach is TkAgg, so a station
    whose python lacks a working tkinter falls back to Agg even with DISPLAY
    set.
    """
    if not os.environ.get("DISPLAY"):
        return None
    try:
        import matplotlib.pyplot as plt

        if plt.get_backend().lower() == "agg":
            return None
        return _Plot(plt, arms)
    except Exception as exc:  # noqa: BLE001 - the event stream is the real job
        print(f"[plot] disabled: {exc}", file=sys.stderr, flush=True)
        return None


class _Plot:
    """Live displacement traces. Best-effort throughout.

    Same contract as `_PreviewWindow` in verify_device_assignment.py: one `_ok`
    kill switch flipped by any failure, every teardown call swallowed, and a
    render that early-returns once disabled. The check's real job is the event
    stream, and a figure that cannot draw must never take it down.
    """

    def __init__(self, plt, arms):
        self._plt = plt
        self._ok = True
        self._closed = False
        self._arms = list(arms)
        self._t = {arm: deque(maxlen=600) for arm in self._arms}
        self._y = {arm: deque(maxlen=600) for arm in self._arms}
        try:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(8, 4))
            # Connect Matplotlib window closure, including its q shortcut, to watch-loop
            # cancellation so polling stops when the plot closes.
            self._fig.canvas.mpl_connect("close_event", self._on_close)
            self._lines = {
                arm: self._ax.plot([], [], label=arm)[0] for arm in self._arms
            }
            self._ax.axhline(MOVE_M, linestyle="--", label=f"moved ({MOVE_M:g} m)")
            self._ax.axhline(STILL_M, linestyle=":", label=f"still ({STILL_M:g} m)")
            self._ax.set_xlabel("seconds")
            self._ax.set_ylabel("movement (m)")
            self._ax.legend(loc="upper left")
            self._fig.canvas.manager.set_window_title("UMI — Vive tracker check")
        except Exception:  # noqa: BLE001
            self._ok = False

    def _on_close(self, _event):
        self._closed = True

    def closed(self) -> bool:
        """True once the operator has closed the window -- by 'q', the title
        bar's close control, or anything else matplotlib treats as closing the
        figure. `watch` polls this the same way it polls `should_stop()`."""
        return self._closed

    def update(self, elapsed, amplitudes, prompted):
        if not self._ok:
            return
        try:
            for arm in self._arms:
                value = amplitudes.get(arm)
                self._t[arm].append(elapsed)
                self._y[arm].append(0.0 if value is None else value)
                self._lines[arm].set_data(list(self._t[arm]), list(self._y[arm]))
            self._ax.set_title(f"Move the {prompted.upper()} gripper")
            self._ax.relim()
            self._ax.autoscale_view()
            self._plt.pause(0.001)
        except Exception:  # noqa: BLE001
            self._ok = False

    def close(self):
        if not self._ok:
            return
        try:
            self._plt.close(self._fig)
        except Exception:  # noqa: BLE001
            pass


def watch(
    sources, out, should_stop, plot=None, *, walk=None, make_client=ViveTrackerClient
) -> int:
    """Walk the operator through one arm at a time. 0 on success, 1 on a cross.

    One client for the whole rig: the bridge's reply already carries every
    tracker, so N sources cost one round trip, and reading them all from one
    sample is what makes the arms comparable.

    `walk` names the arms to confirm, defaulting to all of them. The Station
    Console passes a single arm, because each arm has its own button there and
    the operator starts the next run when ready.

    `walk` does not narrow what gets watched; that is why it is a separate
    argument rather than a shorter `sources` list. A run walking `left` still
    reads the right puck every tick, so the swapped-key cross (the one failure
    nothing downstream can catch) is still found. Given only one source, the
    operator would move the left gripper, the right tracker would move, and the
    check would sit waiting on a still puck until it timed out.

    `make_client` lets the hold state machine below be tested without a live
    bridge, the same seam `verify_device_assignment.watch` has with
    `make_watcher`. Anything with `start`, `stop` and `latest` will do.
    """
    arms = [source["arm"] for source in sources]
    remaining = list(arms) if walk is None else list(walk)

    endpoint = sources[0]["endpoint"]
    client = make_client(endpoint=endpoint, poll_hz=POLL_HZ)
    client.start()

    key_of = {source["arm"]: source["tracker_key"] for source in sources}
    windows = {arm: deque() for arm in arms}
    held_since = None
    last_shape = None
    started = time.monotonic()
    last_emit = 0.0

    _emit(
        out,
        event="ready",
        arms=arms,
        keys=key_of,
        move_m=MOVE_M,
        still_m=STILL_M,
        hold_s=HOLD_S,
    )
    _emit(out, event="prompt", arm=remaining[0])
    print(f"Move the {remaining[0].upper()} gripper around.", flush=True)

    # Treat plot closure as cancellation. Plot-update failures alone are
    # nonfatal, but a closed window must not leave an invisible poller running.
    try:
        while remaining and not should_stop() and not (plot and plot.closed()):
            now = time.monotonic()
            sample = client.latest()
            payload = sample[0] if sample is not None else {}

            amplitudes = {}
            for arm in arms:
                entry = payload.get(key_of[arm]) if isinstance(payload, dict) else None
                # Use the static tracked-flag parser without constructing a ViveTracker,
                # which would create cached clients and background threads.
                tracked = (
                    ViveTracker._tracked_flag(entry)
                    if isinstance(entry, dict)
                    else None
                )
                if not tracked:
                    amplitudes[arm] = None
                    windows[arm].clear()
                    continue
                position = entry.get("position") or {}
                point = (
                    float(position.get("x", 0.0)),
                    float(position.get("y", 0.0)),
                    float(position.get("z", 0.0)),
                )
                window = windows[arm]
                window.append((now, point))
                while window and (now - window[0][0]) > WINDOW_S:
                    window.popleft()
                amplitudes[arm] = amplitude(window)

            prompted = remaining[0]
            # Classify current evidence without dwell time to detect changes in the
            # candidate verdict.
            shape = classify(
                amplitudes,
                prompted=prompted,
                held_s=HOLD_S,
                thresholds=(MOVE_M, STILL_M, HOLD_S),
            )
            # Reset held_since when the evidence changes, before evaluating dwell time.
            # A new candidate verdict must never inherit time accumulated by another
            # one.
            if shape != last_shape:
                last_shape = shape
                held_since = now if shape is not None else None
            verdict = classify(
                amplitudes,
                prompted=prompted,
                held_s=(now - held_since) if held_since is not None else 0.0,
                thresholds=(MOVE_M, STILL_M, HOLD_S),
            )

            if plot is not None:
                plot.update(now - started, amplitudes, prompted)

            if (now - last_emit) >= (1.0 / EMIT_HZ):
                last_emit = now
                _emit(
                    out,
                    event="sample",
                    t=round(now - started, 2),
                    arms={
                        arm: {
                            "disp": None if value is None else round(value, 4),
                            "tracked": value is not None,
                        }
                        for arm, value in amplitudes.items()
                    },
                )

            if verdict == "cross":
                # Use the same >= MOVE_M boundary as classify so a threshold-equal swap
                # always identifies its moving arm.
                moved = next(
                    arm
                    for arm, value in amplitudes.items()
                    if arm != prompted and value is not None and value >= MOVE_M
                )
                _emit(out, event="cross", prompted=prompted, moved=moved)
                print(
                    f"The {prompted.upper()} gripper was moved, but "
                    f"'{key_of[moved]}' is what moved -- '{key_of[prompted]}' did "
                    "not. The tracker keys look swapped; fix them on the "
                    "Configuration page.",
                    file=sys.stderr,
                    flush=True,
                )
                return 1

            if verdict == "confirmed":
                _emit(out, event="confirmed", arm=prompted)
                print(f"{prompted.upper()} arm confirmed.", flush=True)
                remaining.pop(0)
                held_since = None
                last_shape = None
                for window in windows.values():
                    window.clear()
                if remaining:
                    _emit(out, event="prompt", arm=remaining[0])
                    print(f"Now move the {remaining[0].upper()} gripper.", flush=True)

            time.sleep(1.0 / POLL_HZ)
    finally:
        client.stop()
        if plot is not None:
            plot.close()

    # A nonempty remaining set means the check was stopped before every arm
    # confirmed; cancellation is not a device failure.
    return 0 if not remaining else 1


def main():
    parser = argparse.ArgumentParser(
        description="Verify each arm's gripper drives the tracker the config names."
    )
    parser.add_argument("--config", required=True, help="Path to recording config YAML")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Walk each arm in turn and report what moved. Draws no window: the "
        "prompt and the verdict live in the station web UI.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="With --watch, open a live matplotlib figure on the station monitor "
        "showing each tracker's movement. Best-effort: a no-op where there is no "
        "display or no interactive backend.",
    )
    parser.add_argument(
        "--arm",
        help="With --watch, confirm only this arm and return. Every tracker is "
        "still watched, so a swapped pair is still caught. Omit to walk the "
        "whole rig, which is what the bench invocation wants.",
    )
    args = parser.parse_args()

    sources = plan_sources(args.config)
    if not sources:
        print("No Vive trackers in this config.", file=sys.stderr)
        sys.exit(1)
    if not args.watch:
        for source in sources:
            print(f"{source['arm']}: {source['tracker_key']} @ {source['endpoint']}")
        sys.exit(0)

    sys.exit(
        watch(
            sources,
            sys.stdout,
            should_stop=install_stop_on_sigterm(),
            plot=open_plot([s["arm"] for s in sources]) if args.plot else None,
            walk=[args.arm] if args.arm else None,
        )
    )


if __name__ == "__main__":
    main()
