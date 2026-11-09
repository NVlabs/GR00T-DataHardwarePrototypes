# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-facing wording for data-check issues and warnings.

The raw issue strings in ``results["issues"]`` are a stored contract: they are
written to the quarantine attrs and parsed by ``check_dataset``'s
``issues_by_type`` and by tools that read stored reports, so they are never
reworded. This module translates each one, for display only, into what the
operator needs: which check failed, whether it is a HARDWARE issue (a device
misbehaved: check it, swap it if necessary) or a MOTION issue (how the demo
was performed: redo the episode), what happened in plain words, and what to
do.
"""

import re
from dataclasses import dataclass
from typing import List, Sequence

from umi.common.utility.print_utils import Colors

__all__ = [
    "HARDWARE",
    "MOTION",
    "OperatorIssue",
    "explain_issue",
    "overlay_lines",
    "overlay_hint",
    "terminal_lines",
    "warning_lines",
]

HARDWARE = "HARDWARE"
MOTION = "MOTION"

_HEADINGS = {
    MOTION: "MOTION issue (operator) -- redo the episode:",
    HARDWARE: "HARDWARE issue (device) -- check the device; swap it if necessary:",
}
_HEADING_COLORS = {MOTION: Colors.YELLOW, HARDWARE: Colors.PURPLE}


@dataclass(frozen=True)
class OperatorIssue:
    check: str  # data-check name, as keyed in data_check_config.yaml
    kind: str  # HARDWARE or MOTION
    message: str  # what happened, in plain words
    action: str  # what the operator should do
    raw: str  # the original issue string


_DEV = r"(?P<dev>[\w.]+)"

# Motion rules provide a message and action; hardware rules use the common
# device-check action. Format named regex groups into readable device names.
# First match wins, so keep overlapping motion patterns before hardware rules.
# fmt: off
_MOTION_RULES = [
    (rf"^{_DEV}: did not move", "gripper.movement",
     "{dev} never opened or closed",
     "Use the gripper during the task"),
    (rf"^{_DEV}: static", "tracking.pose_movement",
     "{dev} never moved",
     "Move that arm during the task"),
    (rf"^{_DEV}: low consecutive movement", "tracking.consecutive_movement",
     "{dev} was held still for too much of the episode",
     "Keep the arm moving through the task"),
    (rf"^{_DEV}: (required axes did not move|no movement)", "numeric_values",
     "{dev} did not move enough",
     "Move every joint during the task"),
    (rf"^{_DEV}: start frame out of tolerance", "numeric_values",
     "{dev} did not begin in the start pose",
     "Match the start pose before pressing record"),
]
_HARDWARE_RULES = [
    (rf"^{_DEV}: distance jumps detected", "gripper.jumps",
     "{dev} width reading jumped (sensor glitch)"),
    (rf"^{_DEV}: out of range \(\d+ frames\)", "gripper.range",
     "{dev} width reading out of range"),
    (rf"^{_DEV}: pose jumps detected", "tracking.pose_jumps",
     "{dev} tracking jumped (tracking lost)"),
    (rf"^{_DEV}: confirmed tracker excursions", "tracking.excursions",
     "{dev} tracking drifted away and back"),
    (r"^Motion: (?P<dev>\S+) vs (?P<ref>\S+):", "motion_alignment",
     "{dev} and {ref} are out of sync"),
    (rf"^FPS deviation for {_DEV}", "timestamp.fps",
     "{dev} recorded at the wrong frame rate"),
    (rf"^Frame interval violations for {_DEV}", "timestamp.frame_gaps",
     "{dev} dropped frames"),
    (rf"^Sustained frame drop for {_DEV}", "timestamp.frame_drop",
     "{dev} frame rate dropped for a while"),
    (rf"^Non-monotonic timestamps for {_DEV}", "timestamp.monotonic",
     "{dev} clock went backwards"),
    (r"^Low timestamp coverage", "timestamp.coverage",
     "Devices did not all record for the whole episode"),
    (r"^Device consistency: Missing", "device_name_consistency",
     "A device is missing from the recording"),
    (r"^Device consistency:", "device_name_consistency",
     "A device has an unexpected name"),
    (rf"^{_DEV}: resolution", "video_resolution",
     "{dev} recorded at the wrong resolution"),
    (rf"^{_DEV}: (Video file not found|Failed to open video|Error reading video)",
     "video_resolution", "{dev} video is missing or unreadable"),
    (rf"^{_DEV}: (missing|registered but no chunks written)", "numeric_values",
     "{dev} recorded no data"),
    (rf"^{_DEV}: \d+ NaN values", "numeric_values",
     "{dev} sent invalid values"),
    (rf"^{_DEV}: (out of range \[|\d+ jumps detected)", "numeric_values",
     "{dev} reading glitched"),
    (rf"^{_DEV}: shape mismatch", "numeric_values",
     "{dev} sent data in the wrong format"),
    (rf"^{_DEV}: start_within_tolerance", "numeric_values",
     "Data-check config error for {dev}"),
    (r"^Check failed with error", "data_check",
     "The data check itself crashed"),
]
# fmt: on
_COMPILED = [(re.compile(p), c, MOTION, m, a) for p, c, m, a in _MOTION_RULES] + [
    (re.compile(p), c, HARDWARE, m, None) for p, c, m in _HARDWARE_RULES
]


def _readable(name: str) -> str:
    # left_umi_gripper -> "left gripper"; hand.joint_pos -> "hand joint pos"
    return name.replace("umi_", "").replace("_", " ").replace(".", " ")


def _hardware_action(dev: str | None) -> str:
    if dev:
        return f"Check {dev}; swap it if necessary"
    return "Check the devices; swap one if necessary"


def explain_issue(issue: str) -> OperatorIssue:
    for rx, check, kind, message, action in _COMPILED:
        m = rx.match(issue)
        if m:
            groups = {
                k: _readable(v) if k in ("dev", "ref") else v
                for k, v in m.groupdict().items()
            }
            if action is None:
                action = _hardware_action(groups.get("dev"))
            return OperatorIssue(
                check, kind, message.format(**groups), action.format(**groups), issue
            )
    # Unknown wording: show it verbatim rather than guess at a cause.
    return OperatorIssue("data_check", HARDWARE, issue, _hardware_action(None), issue)


def _motion_first(issues: Sequence[str]) -> List[OperatorIssue]:
    # Match motion failures before hardware faults.
    explained = [explain_issue(i) for i in issues]
    return sorted(explained, key=lambda e: e.kind != MOTION)


def terminal_lines(issues: Sequence[str]) -> List[str]:
    """Colored lines for the recorder terminal, grouped by kind.

    Each issue gets one line (check: what happened -> what to do) and a second
    line with the raw check output.
    """
    explained = _motion_first(issues)
    lines: List[str] = []
    for kind in (MOTION, HARDWARE):
        group = [e for e in explained if e.kind == kind]
        if not group:
            continue
        lines.append(f"  {_HEADING_COLORS[kind]}{_HEADINGS[kind]}{Colors.RESET}")
        for e in group:
            lines.append(f"    -> [{e.check}] {e.message}. {e.action}.")
            lines.append(f"       (details: {e.raw})")
    return lines


def warning_lines(warnings: Sequence[str]) -> List[str]:
    """Colored lines for the recorder terminal, for warnings: they never fail
    the episode, so there is nothing to redo -- only something worth a look."""
    if not warnings:
        return []
    lines = [
        f"  {Colors.YELLOW}WARNING (does not fail the episode) -- worth a look:"
        f"{Colors.RESET}"
    ]
    for e in (explain_issue(w) for w in warnings):
        lines.append(f"    -> [{e.check}] {e.message}.")
        lines.append(f"       (details: {e.raw})")
    return lines


def overlay_lines(issues: Sequence[str]) -> List[str]:
    """One entry per issue for the preview's error overlay: a tagged headline
    with the check name, and the action on a second line."""
    return [
        f"[{e.kind}] {e.check}: {e.message}\n-> {e.action}"
        for e in _motion_first(issues)
    ]


def overlay_hint(issues: Sequence[str]) -> str:
    kinds = {explain_issue(i).kind for i in issues}
    if HARDWARE in kinds:
        return "Check the device, swap it if necessary -- press R to resume"
    return "Redo the episode as described above -- press R to resume"
