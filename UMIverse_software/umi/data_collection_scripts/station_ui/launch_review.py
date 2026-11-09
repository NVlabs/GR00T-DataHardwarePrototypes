# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything that is missing or needs redoing before a recording starts.

`readiness.blocking` answers "may Start be pressed": it names the first
outstanding step. This answers the operator's next question, "what else?", so
the Record section can list every missing setting and every step to redo up
front, instead of one refusal per press.

Each item says what is wrong and where it is fixed (`href` is a section anchor
of the single workflow page). `level` is `block` when Start would refuse and
`warn` when recording is allowed but will produce data that may not convert.
"""

from __future__ import annotations

from dataclasses import dataclass

from station_ui import calibration, enforcement, gripper_pairing, identification
from station_ui import devices as device_state
from station_ui.errors import FieldError

BLOCK = "block"
WARN = "warn"


@dataclass(frozen=True)
class Item:
    level: str
    section: str
    message: str
    href: str
    action: str


def _step_items(app, config: dict) -> list[Item]:
    items = []
    try:
        steps = calibration.steps_for(config)
    except FieldError:
        return items
    for step in steps:
        state = calibration.step_state(app, step.id)
        if state == calibration.DONE:
            continue
        if calibration.stale(app, step.id):
            why = "Redo it: the devices changed after it last ran."
        elif state == calibration.FAILED:
            why = "It failed. Fix the problem and run it again."
        elif state == calibration.RUNNING:
            why = "It is still running. Wait for it to finish."
        else:
            why = "It has not been run yet."
        items.append(
            Item(
                BLOCK,
                "Calibration",
                f"{step.title}: {why}",
                "#calibration",
                "Open Calibration",
            )
        )
    return items


def review(
    app,
    config: dict,
    *,
    has_preset: bool,
    config_saved: bool,
    configuration_will_be_saved: bool,
    placeholder_description: str,
) -> list[Item]:
    """Outstanding items for the rig the draft names now, most urgent first."""
    if not has_preset:
        return [Item(BLOCK, "Setup", "No setup chosen yet.", "#setup", "Open Setup")]
    items: list[Item] = []
    if not config_saved and not configuration_will_be_saved:
        items.append(
            Item(
                BLOCK,
                "Configuration",
                "The configuration has not been saved yet.",
                "#configuration",
                "Open Configuration",
            )
        )
    if config:
        meta = config.get("meta") or {}
        if not str(meta.get("task_name") or "").strip():
            items.append(
                Item(
                    BLOCK,
                    "Configuration",
                    "No task name is set.",
                    "#configuration",
                    "Open Configuration",
                )
            )
        description = str(meta.get("task_description") or "").strip()
        if not description:
            items.append(
                Item(
                    BLOCK,
                    "Configuration",
                    "The task has no description.",
                    "#configuration",
                    "Open Configuration",
                )
            )
        elif description == placeholder_description:
            items.append(
                Item(
                    BLOCK,
                    "Configuration",
                    "The task description is still the placeholder.",
                    "#configuration",
                    "Open Configuration",
                )
            )
        found, _ = device_state.outcome_for(device_state.current_job(app))
        for mismatch in gripper_pairing.config_mismatches(config, found):
            items.append(
                Item(
                    BLOCK,
                    "Configuration",
                    str(mismatch),
                    "#configuration",
                    "Open Configuration",
                )
            )
    items += _step_items(app, config or {})
    if identification.running(app):
        items.append(
            Item(
                BLOCK,
                "Setup",
                identification.MESSAGE,
                "#configuration",
                "Open Configuration",
            )
        )
    if calibration.check_running(app):
        items.append(
            Item(
                BLOCK,
                "Calibration",
                device_state.CHECK_HOLDS_THE_CAMERAS,
                "#calibration",
                "Open Calibration",
            )
        )
    warn_only = enforcement.warn_only_titles(config or {})
    if warn_only:
        items.append(
            Item(
                WARN,
                "Recording checks",
                "Warn only: "
                + ", ".join(warn_only)
                + ". Recording will not stop for these, and LeRobot conversion can fail on the data.",
                "#recording-checks",
                "Open Recording checks",
            )
        )
    return items
