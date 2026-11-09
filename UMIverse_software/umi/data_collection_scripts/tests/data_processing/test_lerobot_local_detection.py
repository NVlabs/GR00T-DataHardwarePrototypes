# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local camera detection distinguishes failed alignment from an unsupported rig."""

import pytest

from real_scripts.data_processing.lerobot_conversion.umi_lerobot_conversion import (
    detect_embodiment_tag,
)


def test_metadata_only_episode_reports_failed_alignment(tmp_path):
    episode = tmp_path / "episode_0"
    episode.mkdir()
    (episode / "zarr.json").write_text('{"node_type": "group", "attributes": {}}')
    with pytest.raises(
        ValueError, match="No camera videos.*Alignment is incomplete"
    ) as error:
        detect_embodiment_tag(tmp_path)
    assert "Run Check and Re-align" in str(error.value)


def test_complete_bimanual_camera_set_still_detects_its_embodiment(tmp_path):
    episode = tmp_path / "episode_0"
    episode.mkdir()
    for name in (
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
    ):
        (episode / f"{name}.mp4").touch()
    for name in ("left_tracking_camera", "right_tracking_camera"):
        (episode / name).mkdir()
    assert detect_embodiment_tag(tmp_path) == "gripper_umi_trigger_camera"
