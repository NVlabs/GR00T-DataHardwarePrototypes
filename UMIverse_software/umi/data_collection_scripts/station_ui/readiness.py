# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How far through the workflow this station actually is. One answer.

Two things ask the question and they must never disagree: the numbered tabs in
the shell, which tick each step as it completes, and the Start control on
`/record`, which must be unavailable unless all previous steps passed. Both
read this module. What a step is done by is stated once:

* **Setup**: a preset is chosen. That is the whole of the page.
* **Configuration**: the file the station records from exists. Not "the form
  looks filled in": the artefact, on disk, where `_disarm` can also remove it.
* **Calibration**: every step this rig has passed. Per-rig, not a fixed list:
  only a rig that records Vive trackers is asked for the tracker check. See
  `calibration.steps_for`.

There is no separate device-test step: Configuration already runs the device
scan (`scan_worker --probe`) on load, through `station_ui.devices`, and devices
that cannot be opened are caught again at record time.

Steps 2 and 3 are not asked at all until step 1 is done: they are questions
about a rig the operator has not chosen yet, and answering them would tick
boxes for a session that does not exist.

`blocking` returns the first outstanding step, not all of them, because that is
the one the operator can act on: Configuration cannot be applied before a
setup picks the template it is built from.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from fastapi import FastAPI

from station_ui import calibration, recording
from station_ui import devices as device_state
from station_ui import session as session_state
from station_ui.core.config_assembler import ConfigAssembler, SessionSpec
from station_ui.errors import FieldError

__all__ = [
    "Blocker",
    "blocking",
    "calibration_done",
    "done",
    "generated_config",
    "reachable",
    "reset_verification_for_new_rig",
]


@dataclass(frozen=True)
class Blocker:
    """One outstanding step, phrased for the person who has to clear it.

    `step` is the word on the tab, so what the banner names and what the
    operator looks for in the tab bar are the same string. `message` carries
    the fix rather than only the fault -- "prerequisites not met" is a sentence
    that sends someone hunting.
    """

    href: str
    step: str
    message: str
    action: str


SETUP = Blocker(
    href="/",
    step="Setup",
    message="No setup chosen yet — choose one on Setup.",
    action="Go to Setup",
)
CONFIGURATION = Blocker(
    href="/config",
    step="Configuration",
    message="Complete Configuration. It is saved when you start calibration or recording.",
    action="Go to Configuration",
)
CALIBRATION = Blocker(
    href="/calibration",
    step="Calibration",
    # Describe the outstanding per-rig step without hardcoding the calibration
    # catalogue.
    message="Calibration has not passed — finish the steps on Calibration.",
    action="Go to Calibration",
)


def generated_config(app: FastAPI) -> pathlib.Path | None:
    """Where the current draft's recording config is written, if it has one.

    Asked of the assembler rather than rebuilt from constants, for the same
    reason `station_ui/routes/config.py:_generated_path` does it: the tab may
    not claim Configuration is done against a path Apply does not write.
    """
    draft = app.state.rig.draft
    if not draft.preset_id:
        return None
    try:
        settings = app.state.settings
        return ConfigAssembler(
            settings.repo_root, state_dir=settings.state_dir
        ).output_path(
            SessionSpec(
                preset_id=draft.preset_id,
                operator_name="",
                task_name="",
                session_uid="0" * 12,
                data_dir="",
            )
        )
    except (KeyError, ValueError, OSError):
        # Unknown presets or unreadable templates leave the step incomplete
        # without preventing page rendering.
        return None


def calibration_done(app: FastAPI) -> bool:
    """Every step this rig has passed, against the rig the draft names now.

    Calibration has an observable completion: `calibration.step_state` derives
    done from the children's own exit codes and the camera-check fold, so
    Record waits for it.

    `steps_for` rather than the catalogue: a Vive rig has a tracker check and
    every other rig does not, and requiring the catalogue would block Record on
    every rig behind a check with no trackers to look at.

    Starting a new session removes the old recording config, but the hardware
    checks remain valid. Use the selected rig's roster in that interval; the
    recording action still saves and validates its actual config before launch.
    """
    try:
        config = calibration.layout_config(app)
    except FieldError:
        return False
    steps = calibration.steps_for(config)
    # `bool(steps)`, because `all([])` is True: an empty list must never be the
    # way Record unblocks itself.
    return bool(steps) and all(
        calibration.step_state(app, step.id) == calibration.DONE for step in steps
    )


def _config_written(app: FastAPI) -> bool:
    generated = generated_config(app)
    try:
        return generated is not None and generated.is_file()
    except OSError:
        # A template-read failure leaves readiness incomplete without breaking
        # navigation.
        return False


# Store the last applied rig on app.state rather than in the draft,
# which is serialized into recording config.
_APPLIED = "applied_fingerprint"


def reset_verification_for_new_rig(app: FastAPI) -> bool:
    """Reset everything downstream of the rig when Apply changes it. True if it
    did.

    Called from Apply, and it is the one place a rig change is acted on, so the
    pages behind Configuration cannot keep a different amount of the previous
    rig:

    * **Calibration**: its skip, plus the recorded runs, plus the
      check watcher, which is the only part that touches hardware: left alive it
      holds every camera open for a rig this config no longer names.
    * **Record**: the finished recording the status card still reports, and
      the tick beside it. See `recording.forget_last_recording`.

    Nothing is reset wholesale: each half is gated on the thing it is about, and
    the two questions are genuinely different.

    The verification steps are compared against what they themselves last
    verified. That is what self-heals a change-and-back: swap a gripper, put
    the original back, and the rig is the one that was calibrated again, so the
    result stands rather than costing a fifteen-minute re-run for a net edit of
    nothing. Stamps only exist for a step that has actually run, which is exactly
    right here: a step that never ran has nothing to reset.

    The recording status carries no stamp of its own (a finished child is just
    a job id), so it is compared against the rig the last Apply committed,
    remembered here. An identical re-apply (a task-name typo fixed, say) leaves
    it alone, and so does the first Apply of a process: there is no earlier rig
    for this one to differ from, and nothing has been recorded yet either.
    """
    fingerprint = session_state.device_fingerprint(app.state.rig.draft)
    previous = getattr(app.state, _APPLIED, None)
    setattr(app.state, _APPLIED, fingerprint)

    cal_state = calibration.calibration_state(app)
    stamps = [cal_state.stamp_of(step.id) for step in calibration.STEPS]
    # "" is a step that never ran, which has nothing to expire.
    expired = any(stamp not in ("", fingerprint) for stamp in stamps)
    # Keep stale verification stamps for display, but do not invalidate newly
    # completed steps on every automatic save.
    reset_needed = expired and previous != fingerprint
    if reset_needed:
        calibration.reset(app)
        device_state.reset(app)

    rig_changed = previous is not None and previous != fingerprint
    forgotten = rig_changed and recording.forget_last_recording(app)
    return reset_needed or forgotten


def _steps(app: FastAPI):
    """`(href, done, blocker)` per numbered step, in the operator's order."""
    if not app.state.rig.draft.preset_id:
        # Steps 2 and 3 are questions about a rig that has not been chosen.
        yield "/", False, SETUP
        return
    yield "/", True, SETUP
    yield "/config", _config_written(app), CONFIGURATION
    yield "/calibration", calibration_done(app), CALIBRATION


def done(app: FastAPI) -> frozenset[str]:
    """The hrefs whose step is complete -- what the tab bar ticks."""
    return frozenset(href for href, complete, _ in _steps(app) if complete)


def blocking(
    app: FastAPI, *, configuration_will_be_saved: bool = False
) -> Blocker | None:
    """The first step still outstanding, or None when the station is ready.

    The Start control submits Configuration, so it may account for that pending
    save. The launch gate uses the default and requires the file to exist after
    saving and validating it. Calibration is required in both cases.
    """
    for href, complete, blocker in _steps(app):
        if configuration_will_be_saved and href == "/config":
            continue
        if not complete:
            return blocker
    return None


def reachable(app: FastAPI) -> frozenset[str]:
    """The hrefs an operator may open: every done step, plus the first one that
    is not.

    This is the "otherwise, forbidden" half of the sequential workflow. A step
    past the first gap is a page whose questions have no answers yet (the
    calibration of devices nobody has confirmed are plugged in), and the
    console should not answer those.

    `/record` and `/audit` are not here. Record is gated by `blocking` at the
    point of action, which is stricter than a nav rule and already refuses;
    Data Audit is about episodes that were recorded in the past and has nothing
    to do with how far through this session the operator is.
    """
    allowed = set()
    for href, complete, _blocker in _steps(app):
        allowed.add(href)
        if not complete:
            break
    return frozenset(allowed)
