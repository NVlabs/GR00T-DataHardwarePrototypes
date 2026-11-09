# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which recorded groups carry a tracked pose, and which arm each one drives.

One definition, shared by everything that reads a recorded episode, so every
consumer resolves pose sources the same way. A consumer that cannot resolve a
tracking device skips its check without raising, so a separate list per
consumer would fail silently whenever a new pose source was added.

The naming convention already encodes the answer, so these are rules rather than
tables and need no maintenance as sources are added.
"""

from typing import List, Optional, Tuple

__all__ = [
    "ARM_SIDES",
    "tracking_devices_in",
    "tracking_device_to_arm",
    "tracking_device_side",
    "tracking_device_for_side",
]

ARM_SIDES: Tuple[str, ...] = ("left", "right")


def tracking_devices_in(devices: List[str]) -> List[str]:
    """Every device group that carries a tracked pose.

    The rule is the substring "tracking" in the group name, and it is the only
    rule: it stays agnostic about what produced the pose. An iPhone rig
    records ``{side}_tracking_camera`` (ARKit pose inside a camera frame); a
    Vive rig records ``{side}_tracking_pose`` (a pose-only numeric source).
    Both are tracking devices and every consumer treats them identically.
    """
    return [d for d in devices if "tracking" in d.lower()]


def tracking_device_to_arm(device_name: str) -> Optional[str]:
    """Which arm a tracking group drives, or None if it drives none.

    Derived from the leading side token: ``{side}_...`` -> ``{side}_arm``.

    ``head_tracking_camera`` correctly resolves to None -- the head is not an
    arm, and it must keep returning None.
    """
    side = device_name.split("_", 1)[0].lower()
    return f"{side}_arm" if side in ARM_SIDES else None


def tracking_device_side(device_name: str) -> Optional[str]:
    """The bare side a source name leads with, or None if it names no arm.

    The same rule as :func:`tracking_device_to_arm` and defined in terms of it
    so the two cannot drift, for callers that want ``"left"`` rather than
    ``"left_arm"``, such as the Station Console's Vive rows, form fields and
    per-arm prompts.

    ``head_tracking_pose`` returns None, exactly as it does above: a puck that
    names no arm is skipped rather than given an invented one.
    """
    arm = tracking_device_to_arm(device_name)
    return arm[: -len("_arm")] if arm is not None else None


def tracking_device_for_side(side: str, candidates: List[str]) -> Optional[str]:
    """The tracking group belonging to ``side``, or None if undecidable.

    Inverse of :func:`tracking_device_to_arm`, and defined in terms of it so the
    two cannot drift. Callers that know a side (from a gripper name) use this to
    find the pose that goes with it. An ambiguous side -- two tracking groups
    for one arm -- returns None rather than guessing.
    """
    want = f"{side}_arm"
    matches = [d for d in candidates if tracking_device_to_arm(d) == want]
    return matches[0] if len(matches) == 1 else None
