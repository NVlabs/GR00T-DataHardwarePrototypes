# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that create_video_recorders() returns flat expanded configs for all source types.

Verifies that both individual-source configs (oak_camera, iphone_camera, mock_camera)
and multi-stream configs (oak_device with nested streams) produce expanded config
entries where every entry has a top-level "name" key.

Also verifies that downstream align_data can parse the saved config.yaml.
"""

import os
from unittest.mock import MagicMock, patch

import yaml

from real_scripts.data_processing.align_data import _config_source_names
from umi.common.utility.recording_utils import create_video_recorders


def _make_mock_source(device_name: str) -> MagicMock:
    """Create a minimal mock camera source."""
    source = MagicMock()
    source.device_name = device_name
    return source


class TestExpandedVideoConfigs:
    """Verify create_video_recorders returns flat expanded_configs."""

    def test_individual_sources_have_name(self, tmp_path):
        """Config style: one camera per source -- each source has a name."""
        video_source_configs = [
            {
                "type": "mock_camera",
                "name": "left_umi_camera",
                "frame_type": "RGBCameraFrame",
                "latency": 0.0875,
                "record_fps": 45,
                "stream_fps": 60,
                "params": {"camera_resolution": [640, 400], "fps": 60},
            },
            {
                "type": "mock_camera",
                "name": "right_umi_camera",
                "frame_type": "RGBCameraFrame",
                "latency": 0.0875,
                "record_fps": 45,
                "stream_fps": 60,
                "params": {"camera_resolution": [640, 400], "fps": 60},
            },
        ]

        _, _, expanded = create_video_recorders(
            video_source_configs=video_source_configs,
            record_path=str(tmp_path),
        )

        assert len(expanded) == 2
        for entry in expanded:
            assert "name" in entry, f"Missing 'name' key in expanded config: {entry}"
        names = {e["name"] for e in expanded}
        assert names == {"left_umi_camera", "right_umi_camera"}

    def test_oak_device_streams_expanded_to_flat(self, tmp_path):
        """Config style: bimanual_oak_ffc_iphone_*_config.yaml — oak_device with streams."""
        video_source_configs = [
            {
                "type": "oak_device",
                "oak_device_name": "oak-FFC-03",
                "fps": 45,
                "latency": 0.0823,
                "streams": [
                    {
                        "name": "left_trigger_camera",
                        "socket": "CAM_A",
                        "resolution": [640, 400],
                    },
                    {
                        "name": "left_umi_camera",
                        "socket": "CAM_B",
                        "resolution": [640, 400],
                    },
                ],
            },
            {
                "type": "mock_camera",
                "name": "right_tracking_camera",
                "frame_type": "RGBCameraFrame",
                "latency": 0.13,
                "record_fps": 45,
                "stream_fps": 60,
                "params": {"camera_resolution": [640, 400], "fps": 60},
            },
        ]

        mock_streams = [
            _make_mock_source("left_trigger_camera"),
            _make_mock_source("left_umi_camera"),
        ]

        with (
            patch(
                "umi.common.utility.recording_utils.create_video_source"
            ) as mock_create,
            patch("umi.common.utility.recording_utils.VideoRecorder"),
        ):

            def side_effect(cfg):
                if cfg["type"] == "oak_device":
                    return mock_streams
                return [_make_mock_source(cfg["name"])]

            mock_create.side_effect = side_effect

            _, _, expanded = create_video_recorders(
                video_source_configs=video_source_configs,
                record_path=str(tmp_path),
            )

        assert len(expanded) == 3
        for entry in expanded:
            assert "name" in entry, f"Missing 'name' key in expanded config: {entry}"

        names = {e["name"] for e in expanded}
        assert names == {
            "left_trigger_camera",
            "left_umi_camera",
            "right_tracking_camera",
        }

    def test_expanded_configs_contain_required_fields(self, tmp_path):
        """Every expanded entry must have the fields downstream code expects."""
        video_source_configs = [
            {
                "type": "mock_camera",
                "name": "test_camera",
                "latency": 0.05,
                "record_fps": 30,
                "stream_fps": 60,
                "params": {"camera_resolution": [320, 240], "fps": 30},
            },
        ]

        _, _, expanded = create_video_recorders(
            video_source_configs=video_source_configs,
            record_path=str(tmp_path),
        )

        assert len(expanded) == 1
        entry = expanded[0]
        required_fields = {
            "name",
            "type",
            "frame_type",
            "latency",
            "record_fps",
            "stream_fps",
        }
        assert required_fields.issubset(entry.keys()), (
            f"Missing fields: {required_fields - entry.keys()}"
        )
        assert entry["name"] == "test_camera"
        assert entry["type"] == "mock_camera"
        assert entry["latency"] == 0.05
        assert entry["record_fps"] == 30
        assert entry["stream_fps"] == 60

    def test_oak_device_inherits_parent_latency(self, tmp_path):
        """oak_device streams without their own latency inherit from the parent entry."""
        video_source_configs = [
            {
                "type": "oak_device",
                "oak_device_name": "oak-FFC-01",
                "fps": 45,
                "latency": 0.0823,
                "streams": [
                    {"name": "cam_a", "socket": "CAM_A", "resolution": [640, 400]},
                ],
            },
        ]

        with patch(
            "umi.common.utility.recording_utils.create_video_source"
        ) as mock_create:
            mock_create.return_value = [_make_mock_source("cam_a")]

            _, _, expanded = create_video_recorders(
                video_source_configs=video_source_configs,
                record_path=str(tmp_path),
            )

        assert len(expanded) == 1
        assert expanded[0]["latency"] == 0.0823


class TestAlignDataParsesExpandedConfig:
    """Verify align_data._config_source_names works with expanded configs saved to disk."""

    def _write_config(self, data_dir, video_sources, numeric_sources=None):
        """Write a config.yaml with given sources into data_dir."""
        os.makedirs(data_dir, exist_ok=True)
        config = {"video_sources": video_sources}
        if numeric_sources:
            config["numeric_sources"] = numeric_sources
        with open(os.path.join(data_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f)

    def test_individual_sources_parsed(self, tmp_path):
        """Flat expanded configs from bimanual_umi_gripper style are parsed correctly."""
        expanded = [
            {
                "name": "left_umi_camera",
                "type": "oak_camera",
                "frame_type": "RGBCameraFrame",
            },
            {
                "name": "right_umi_camera",
                "type": "oak_camera",
                "frame_type": "RGBCameraFrame",
            },
        ]
        numeric = [
            {
                "name": "left_umi_gripper",
                "type": "umi_gripper",
                "frame_type": "GripperFrame",
            },
        ]
        data_dir = str(tmp_path / "dataset")
        self._write_config(data_dir, expanded, numeric)

        names = _config_source_names(data_dir)
        assert names == {"left_umi_camera", "right_umi_camera", "left_umi_gripper"}

    def test_oak_device_expanded_configs_parsed(self, tmp_path):
        """Flat expanded configs from oak_device (after expansion) are parsed correctly."""
        expanded = [
            {
                "name": "left_trigger_camera",
                "type": "oak_device",
                "frame_type": "RGBCameraFrame",
            },
            {
                "name": "left_umi_camera",
                "type": "oak_device",
                "frame_type": "RGBCameraFrame",
            },
            {
                "name": "right_tracking_camera",
                "type": "iphone_camera",
                "frame_type": "RGBCameraFrame",
            },
        ]
        data_dir = str(tmp_path / "dataset")
        self._write_config(data_dir, expanded)

        names = _config_source_names(data_dir)
        assert names == {
            "left_trigger_camera",
            "left_umi_camera",
            "right_tracking_camera",
        }

    def test_end_to_end_expand_then_parse(self, tmp_path):
        """Full round-trip: create_video_recorders expands, save to disk, align_data parses."""
        video_source_configs = [
            {
                "type": "oak_device",
                "oak_device_name": "oak-FFC-03",
                "fps": 45,
                "latency": 0.0823,
                "streams": [
                    {
                        "name": "left_trigger_camera",
                        "socket": "CAM_A",
                        "resolution": [640, 400],
                    },
                    {
                        "name": "left_umi_camera",
                        "socket": "CAM_B",
                        "resolution": [640, 400],
                    },
                ],
            },
            {
                "type": "mock_camera",
                "name": "right_tracking_camera",
                "frame_type": "RGBCameraFrame",
                "latency": 0.13,
                "record_fps": 45,
                "stream_fps": 60,
                "params": {"camera_resolution": [640, 400], "fps": 60},
            },
        ]

        with patch(
            "umi.common.utility.recording_utils.create_video_source"
        ) as mock_create:

            def side_effect(cfg):
                if cfg["type"] == "oak_device":
                    return [_make_mock_source(s["name"]) for s in cfg["streams"]]
                from umi.common.hardware_sdk.camera.mock_camera import MockCamera

                return [MockCamera(device_name=cfg["name"], **cfg.get("params", {}))]

            mock_create.side_effect = side_effect

            record_dir = str(tmp_path / "recording")
            os.makedirs(record_dir, exist_ok=True)
            _, _, expanded = create_video_recorders(
                video_source_configs=video_source_configs,
                record_path=record_dir,
            )

        # Simulate what record_data.py does: save expanded config
        data_dir = str(tmp_path / "saved_dataset")
        self._write_config(data_dir, expanded)

        # Verify align_data can parse it
        names = _config_source_names(data_dir)
        assert names == {
            "left_trigger_camera",
            "left_umi_camera",
            "right_tracking_camera",
        }
