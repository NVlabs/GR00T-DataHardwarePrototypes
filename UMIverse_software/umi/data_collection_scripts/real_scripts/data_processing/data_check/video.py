# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video resolution check."""

import os
from typing import Any, Dict, Tuple

__all__ = ["check_video_resolution"]


def check_video_resolution(
    video_path: str, expected_resolution: Tuple[int, int] = (1280, 720)
) -> Dict[str, Any]:
    """Check if video file has the expected resolution.

    Args:
        video_path: Path to video file
        expected_resolution: Expected resolution as (width, height) tuple

    Returns:
        Dict with resolution check results
    """
    import cv2

    result = {
        "passed": False,
        "expected_resolution": expected_resolution,
        "actual_resolution": None,
        "error": None,
    }

    if not os.path.exists(video_path):
        result["error"] = f"Video file not found: {video_path}"
        return result

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result["error"] = f"Failed to open video: {video_path}"
            return result

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        result["actual_resolution"] = (width, height)
        result["passed"] = (width, height) == expected_resolution
    except Exception as e:
        result["error"] = f"Error reading video: {str(e)}"

    return result
