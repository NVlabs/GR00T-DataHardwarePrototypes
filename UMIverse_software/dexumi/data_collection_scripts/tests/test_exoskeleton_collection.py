# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection keeps working with current and older calibration files."""

import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

from umi.common.data.data_buffer import DataBuffer
from umi.common.recording.session_slot import device_config_entries
from umi.umi_device.dexumi_exoskeleton import DexUmiExoskeleton

TEMPLATE_DIR = (
    Path(__file__).resolve().parents[1] / "config/dexumi_exoskeleton/templates"
)
LEGACY_METADATA = (
    Path(__file__).with_name("fixtures") / "legacy_calibration_metadata.yaml"
)


def _saved_calibration(side):
    """A station's calibration of one glove: its side's design template, the zeros
    measured on the glove, and the metadata older calibrations were saved with."""
    config = yaml.safe_load((TEMPLATE_DIR / f"DEXUMI-{side}.template.yaml").read_text())
    config["zero_position_angle"] = [12.5 * joint for joint in range(15)]
    config.update(yaml.safe_load(LEGACY_METADATA.read_text()))
    return config


class ExoskeletonCollectionTest(unittest.TestCase):
    def test_old_calibration_keeps_encoder_and_tactile_recording(self):
        for side in ("Left", "Right"):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as home:
                path = Path(home) / f"DEXUMI-{side}.yaml"
                path.write_text(yaml.safe_dump(_saved_calibration(side)))

                device = DexUmiExoskeleton.from_config(path, reader=None)
                self.assertEqual(
                    [source.device_name for source in device.get_data_sources()],
                    ["dexumi_exoskeleton_encoder", "dexumi_exoskeleton_tactile"],
                )
                raw_angles = device.zero_position_angle + 2
                frame = SimpleNamespace(
                    position=raw_angles, capture_time=1.0, receive_time=1.01
                )
                with mock.patch.object(device.encoder, "get_frame", return_value=frame):
                    recorded = device._get_encoder_frame()
                np.testing.assert_allclose(
                    recorded.joint_angles, 2 * device.direction_signs
                )
                np.testing.assert_array_equal(recorded.raw_encoder_angles, raw_angles)
                self.assertEqual(recorded.capture_time, 1.0)
                self.assertEqual(recorded.receive_time, 1.01)
                tactile_fields = {
                    f.name for f in dataclasses.fields(device.tactile.frame_class)
                }
                self.assertEqual(
                    tactile_fields - {"capture_time", "receive_time", "frame_source"},
                    {
                        f"tactile_{sensor}_{axis}_{point}"
                        for sensor in (33, 34, 35, 36, 37)
                        for axis in ("fx", "fy", "fz")
                        for point in (1, 2)
                    },
                )

    def test_unknown_configuration_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as home:
            config = _saved_calibration("Right")
            config["unknown_setting"] = True
            path = Path(home) / "calibration.yaml"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(TypeError, "unknown_setting"):
                DexUmiExoskeleton.from_config(path, reader=None)

    def test_robot_actions_use_calibration_and_joint_limits(self):
        for side in ("Left", "Right"):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as home:
                path = Path(home) / f"DEXUMI-{side}.yaml"
                path.write_text(yaml.safe_dump(_saved_calibration(side)))
                values = np.zeros((2, 15))
                values[1, 3] = (
                    200  # The calibrated abduction joint is capped at 60 deg.
                )
                actions = DexUmiExoskeleton.convert_to_robot_actions(
                    values, umi_config=path
                )
                self.assertEqual(actions.shape, (2, 16))
                self.assertAlmostEqual(np.rad2deg(actions[1, 3]), 60.0)


class CalibrationBundlingTest(unittest.TestCase):
    def test_recording_and_alignment_keep_calibration_and_provenance(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home)
            calibration = root / "DEXUMI-Right.yaml"
            calibration.write_text(LEGACY_METADATA.read_text())
            config = {
                "numeric_sources": [
                    {"name": name, "params": {"config_file": calibration.name}}
                    for name in ("right_encoder", "right_tactile")
                ]
            }
            (root / "config.yaml").write_text(yaml.safe_dump(config))
            (root / "device_binding.yaml").write_text("{}\n")
            self.assertEqual(
                device_config_entries(config, base_dir=home),
                [(str(calibration), calibration.name)],
            )

            buffer = DataBuffer.__new__(DataBuffer)
            buffer.data_dir = home
            basenames = buffer._bundled_config_basenames()
            self.assertEqual(
                set(basenames),
                {"config.yaml", "device_binding.yaml", calibration.name},
            )
            aligned = root / "aligned"
            aligned.mkdir()
            buffer._copy_bundled_configs(home, str(aligned), basenames)
            for name in basenames:
                self.assertEqual(
                    (aligned / name).read_bytes(), (root / name).read_bytes()
                )


if __name__ == "__main__":
    unittest.main()
