# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UMI recording checks, using DexUMI's enforced / warn-only policy."""

FORM_MARKER = "recording_checks_form"
FORM_FIELD = "enforce."
CHECKS = (
    (
        "pre",
        "umi_device.uart_connected",
        "Gripper connections",
        "Every gripper's USB connection is still open.",
    ),
    (
        "pre",
        "camera.relative_calibration",
        "Camera calibration",
        "The cameras have a valid relative calibration.",
    ),
    ("data", "timestamp", "Frame timing", "Frame rate, coverage and dropped frames."),
    (
        "data",
        "gripper",
        "Gripper readings",
        "Gripper movement, jumps and travel range, as configured for the task.",
    ),
    (
        "data",
        "tracking",
        "Pose tracking",
        "Each iPhone or Vive tracker moved enough and did not jump.",
    ),
    (
        "data",
        "device_name_consistency",
        "Left/right pairing",
        "The recorded devices match the configured camera and gripper roles.",
    ),
    (
        "data",
        "video_resolution",
        "Video resolution",
        "Each video has the expected resolution.",
    ),
    (
        "data",
        "motion_alignment",
        "Camera and sensor motion",
        "The tracking pose follows the camera's motion.",
    ),
)
KEYS = frozenset(f"{stage}:{key}" for stage, key, _, _ in CHECKS)


def catalog():
    return [
        dict(id=f"{stage}:{key}", stage=stage, title=title, detail=detail)
        for stage, key, title, detail in CHECKS
    ]


def validate(relaxed):
    unknown = set(relaxed) - KEYS
    if unknown:
        raise ValueError(f"Unknown recording checks: {', '.join(sorted(unknown))}")
    return sorted(set(relaxed))


def from_form(form):
    values = [name[len(FORM_FIELD) :] for name in form if name.startswith(FORM_FIELD)]
    validate(values)
    return sorted(KEYS - set(values))


def policy(relaxed):
    out = {}
    for item in validate(relaxed):
        stage, key = item.split(":", 1)
        out.setdefault(stage, {})[key] = False
    return out


def restored(config):
    choices = config.get("check_enforcement") or {}
    if not isinstance(choices, dict):
        return []
    return sorted(
        f"{stage}:{key}"
        for stage in ("pre", "data")
        for key, enforced in (
            choices.get(stage) if isinstance(choices.get(stage), dict) else {}
        ).items()
        if enforced is False and f"{stage}:{key}" in KEYS
    )


def warn_only_titles(config):
    """Titles of the checks the saved config runs warn-only, in catalog order."""
    relaxed = set(restored(config))
    return [title for stage, key, title, _ in CHECKS if f"{stage}:{key}" in relaxed]
