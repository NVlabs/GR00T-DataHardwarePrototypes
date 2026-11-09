# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which `video_sources` name carries which QR-verification role.

One table, read by two processes. The web server derives the role list the
check panel is keyed on; the `--watch` worker resolves each of those roles to a
camera. Two copies could drift, and a camera's answer would then be thrown away
under a green `✓ Done`.

Deliberately its own module with no imports. `station_ui/calibration.py` may
not import the worker (it pulls cv2 and every camera SDK at module scope, none
of which belongs in a web process), so the shared fact has to live somewhere
neither of them owns.
"""

from __future__ import annotations

__all__ = ["HEAD_ROLES", "ROLE_SOURCE_NAMES"]

# Camera-role aliases are ordered left, right, then head. Accept configured
# head-camera names and their persisted aliases. The bottom camera records
# data but intentionally has no verification role.
ROLE_SOURCE_NAMES: dict[str, tuple[str, ...]] = {
    "left_tracking": ("left_tracking_camera",),
    "left_umi": ("left_umi_camera",),
    "left_trigger": ("left_trigger_camera",),
    "right_tracking": ("right_tracking_camera",),
    "right_umi": ("right_umi_camera",),
    "right_trigger": ("right_trigger_camera",),
    "head": ("head_iphone_camera", "head_tracking_camera", "head_mount_iphone"),
    "head_umi": ("head_oak_camera", "head_umi_camera"),
}

# Verify a head camera only when meta.use_head_mount enables it for this session.
HEAD_ROLES: tuple[str, ...] = ("head", "head_umi")
