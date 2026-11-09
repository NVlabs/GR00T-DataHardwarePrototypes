# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration utility functions shared across GUI modules."""

import re

from umi.common.utility.output_layout import session_dir


def sanitize_task_name(raw_task: str) -> str:
    """Convert a free-form task name into a filename-safe slug.

    Converts spaces to underscores, strips characters that are not
    alphanumeric, underscores, or hyphens, and collapses consecutive
    underscores/hyphens.

    Examples:
        "Lego Disassembly Toss v2" -> "Lego_Disassembly_Toss_v2"
        "pick & place!" -> "pick_place"
    """
    if not raw_task:
        return "unnamed"
    slug = re.sub(r"\s+", "_", raw_task.strip())
    slug = re.sub(r"[^\w\-]", "", slug)
    slug = re.sub(r"[_\-]{2,}", "_", slug)
    slug = slug.strip("_-")
    return slug or "unnamed"


def build_data_dir(
    date: str,
    task_name: str,
    mode: str,
    right_gripper_num=None,
    left_gripper_num=None,
    camera_type: str = "oak",
    *,
    pose_tracker: str | None = None,
    session_uid: str,
) -> str:
    """Build the full data directory path that matches the project convention.

    Sessions live under the output folder (``umi.common.utility.output_layout``)
    as ``<output>/raw/<task>/<session>``, with ``<trk>`` the preset's pose
    tracker (``iphone`` / ``vive`` / ``notracker``) omitted entirely when the
    caller supplies none:
        Bimanual OAK : <output>/raw/<task>/YYYY-MM-DD_umi<L>L_umi<R>R_<trk>_<task>_<uid>
        Bimanual ELP : <output>/raw/<task>/YYYY-MM-DD_umi<L>L_umi<R>R_<trk>_ELP_<task>_<uid>
        Unimanual OAK: <output>/raw/<task>/YYYY-MM-DD_umi<R>R_<trk>_<task>_<uid>
        Unimanual ELP: <output>/raw/<task>/YYYY-MM-DD_umi<R>R_<trk>_ELP_<task>_<uid>

        Versioned grippers use tokens like V4_0001 so paths look like
        umiV4_0001L (without a duplicated UMI prefix).

    Args:
        date: ISO date string, e.g. "2026-02-05". Required to be exactly
            ``YYYY-MM-DD``: it sits at offset 0, and the audit finds a day's
            sessions by that prefix.
        task_name: Free-form task name; sanitized here into a path-safe slug
            (``sanitize_task_name``) before it enters the folder name.
        mode: "unimanual" or "bimanual".
        right_gripper_num: Gripper number for the right arm (string or int).
        left_gripper_num: Gripper number for the left arm (string or int,
            bimanual only).
        camera_type: "oak" (default) or "elp".
        pose_tracker: The wrist pose source this rig records, from
            ``SetupPreset.pose_tracker``. Omitted from the name when None or
            blank, which is what the CLI path (no preset) gets.
        session_uid: Required 12-char lowercase-hex session id. It is
            keyword-only and has no default because an unsuffixed name can
            collide with another session under the same
            ``raw/{task}/{session}/`` folder and interleave its episodes.

    Returns:
        Absolute path string ``<output>/raw/<task>/<session>``.

    Raises:
        ValueError: if ``session_uid`` is not 12 lowercase hex characters, or
            if ``date`` is not a ``YYYY-MM-DD`` string.
    """
    from station_ui.core.session_identity import build_session_dir_name

    camera_prefix = "ELP_" if camera_type == "elp" else ""
    # Sanitize task text before using it as a path component to prevent nested
    # paths or traversal outside the output root. Blank tasks omit the segment.
    raw_task = "" if task_name is None else str(task_name).strip()
    task_slug = sanitize_task_name(raw_task) if raw_task else ""
    dir_name = build_session_dir_name(
        date=date,
        task_name=f"{camera_prefix}{task_slug}" if task_slug else "",
        mode=mode,
        right_gripper=right_gripper_num,
        left_gripper=left_gripper_num,
        pose_tracker=pose_tracker,
        session_uid=session_uid,
    )
    return session_dir("raw", task_slug, dir_name)
