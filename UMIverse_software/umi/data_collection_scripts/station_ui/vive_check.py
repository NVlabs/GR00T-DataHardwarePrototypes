# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Vive tracker check, as the panel sees it.

Its own module rather than more of `station_ui/calibration.py`, which already
carries the step catalogue, the camera fold and the gripper plumbing. Same
shape as the camera fold next door: parse the child's prefixed lines, fold
them into per-arm state, and derive everything on each call so a row can never
read confirmed from a check that is no longer running.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from station_ui.calibration import parse_events

__all__ = [
    "CONFIRMED",
    "MOVING",
    "STILL",
    "UNTRACKED",
    "VIVE_PREFIX",
    "WAITING",
    "ViveRow",
    "ViveView",
    "fold",
    "parse",
]

# Match verify_vive_assignment.VIVE_PREFIX without importing its ZMQ and
# plotting dependencies. Keep it distinct from the camera-check event prefix.
VIVE_PREFIX = "##station-ui-vive## "

WAITING = "waiting"
MOVING = "moving"
STILL = "still"
CONFIRMED = "confirmed"
UNTRACKED = "untracked"

# Used only when the child's `ready` line is not in the buffer. The job buffer
# trims from the front, so a long enough run evicts its own opening lines,
# and a broken rig would then read back as "never started".
DEFAULT_MOVE_M = 0.05
DEFAULT_STILL_M = 0.01


@dataclass
class ViveRow:
    arm: str
    tracker_key: str = ""
    disp: float | None = None
    state: str = WAITING
    # How full this arm's bar is, 0-100. Filled in by `fold`, which is the only
    # place that knows `move_m`, so the template carries no arithmetic and no
    # second copy of the thresholds.
    percent_value: float = 0.0


@dataclass
class ViveView:
    rows: list = field(default_factory=list)
    prompted: str | None = None
    thresholds: dict = field(
        default_factory=lambda: {
            "move_m": DEFAULT_MOVE_M,
            "still_m": DEFAULT_STILL_M,
        }
    )
    done: bool = False
    cross: dict | None = None
    running: bool = False
    failed: bool = False
    # Jobs encode timeouts as FAILED with a reason string; present them as
    # stopped checks rather than hardware failures.
    timed_out: bool = False
    error: str | None = None
    job_id: str | None = None
    log: list = field(default_factory=list)


def parse(lines) -> list[dict]:
    """Every Vive event the child printed, oldest first."""
    return parse_events(lines, prefix=VIVE_PREFIX)


def fold(events, seed=None, only=None) -> ViveView:
    """Per-arm state from the child's events. Latest wins, confirmed latches.

    `seed` is the arm -> tracker key mapping the config declares, and it is what
    the panel has to show before the check has ever run. Without it the first
    render of a Vive rig's own step would say "No trackers to check" directly
    above the button offering to check them, on a rig that has the step
    *because* it records trackers. The camera check seeds its rows the same way
    from `verification_roles(config)`. It also puts the tracker key on screen to
    eyeball before Start, which matters because SteamVR can rename pucks between
    days.

    `only` is the arms this run was launched to confirm: one, for a console
    that gives each arm its own button. It narrows two things and nothing else:
    which rows come back, and which arms `done` is asked about. A left-only run
    still watches the right puck (that is what catches a swapped pair, and the
    child is careful to keep doing it), but a panel section for the left arm
    must not draw the right one's bar, and must not wait for the right arm to
    confirm before it will call itself done.
    """
    view = ViveView()
    seeded = {str(arm): str(key) for arm, key in (seed or {}).items()}
    arms: list[str] = list(seeded)
    keys: dict[str, str] = dict(seeded)
    disp: dict[str, float | None] = {}
    tracked: dict[str, bool] = {}
    confirmed: set[str] = set()

    for event in events:
        kind = event.get("event")
        if kind == "ready":
            # Merge nonempty child-reported arm keys into the configured seed.
            # Malformed ready events must not clear an otherwise valid panel.
            named = [str(a) for a in event.get("arms") or []]
            if named:
                arms = named
            keys.update({str(k): str(v) for k, v in (event.get("keys") or {}).items()})
            for name in ("move_m", "still_m"):
                # Validate move_m as a positive number before using it as a divisor.
                # Malformed subprocess events must not break the shared page context.
                value = event.get(name)
                if isinstance(value, (int, float)) and value > 0:
                    view.thresholds[name] = float(value)
        elif kind == "prompt":
            view.prompted = event.get("arm")
        elif kind == "confirmed":
            # Latches. The operator moves the other gripper next and this arm's
            # tracker will show motion again (a bench is not still), and an
            # arm that already proved itself must not silently un-confirm.
            confirmed.add(str(event.get("arm")))
        elif kind == "cross":
            view.cross = {
                "prompted": event.get("prompted"),
                "moved": event.get("moved"),
            }
        elif kind == "sample":
            for arm, entry in (event.get("arms") or {}).items():
                if not isinstance(entry, dict):
                    continue
                # Accept only numeric displacements. Treat malformed readings as
                # unavailable
                # rather than raising during threshold comparison.
                reading = entry.get("disp")
                disp[str(arm)] = (
                    float(reading) if isinstance(reading, (int, float)) else None
                )
                tracked[str(arm)] = bool(entry.get("tracked"))
                if str(arm) not in arms:
                    arms.append(str(arm))

    # Filter arms before computing done so a per-arm panel is judged only
    # against the arms it displays.
    if only is not None:
        wanted = set(only)
        arms = [arm for arm in arms if arm in wanted]

    still_m = view.thresholds["still_m"]
    move_m = view.thresholds["move_m"]
    for arm in arms:
        value = disp.get(arm)
        if arm in confirmed:
            state = CONFIRMED
        elif arm in tracked and not tracked[arm]:
            state = UNTRACKED
        elif value is None:
            state = WAITING
        elif value >= move_m:
            state = MOVING
        elif value <= still_m:
            state = STILL
        else:
            # Between the thresholds: deliberate dead space, and honestly
            # neither. "Waiting" is the word for "no verdict yet".
            state = WAITING
        view.rows.append(
            ViveRow(
                arm=arm,
                tracker_key=keys.get(arm, ""),
                disp=value,
                state=state,
                # Clamped, so a gripper swung well past the threshold draws a
                # full bar rather than one running off the panel.
                percent_value=(
                    0.0 if value is None else min(100.0, 100.0 * value / move_m)
                ),
            )
        )

    # Cross-talk always overrides confirmation: swapped tracker keys must
    # never produce a verified rig.
    view.done = bool(arms) and not view.cross and all(arm in confirmed for arm in arms)
    return view
