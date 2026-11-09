# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

import gear_neck.load_config as loader
from gear_neck.load_config import ConfigError, Ids, load_config, resolve_config

DEFAULT = yaml.safe_load(loader.DEFAULT_CONFIG_PATH.read_text())


class DefaultFileTest(unittest.TestCase):
    def test_loads_the_neck_settings(self) -> None:
        settings = load_config()

        self.assertEqual([joint.name for joint in settings.joints], ["yaw", "roll", "pitch"])
        self.assertEqual(
            settings.neck_ids, {"yaw": Ids(50, 51), "roll": Ids(52, 53), "pitch": Ids(54, 55)}
        )
        # The yaw drive is mounted reversed.
        self.assertEqual([joint.direction for joint in settings.joints], [-1.0, 1.0, 1.0])
        self.assertEqual(settings.joints_by_name["pitch"].max_deg, 80.0)
        self.assertEqual([(joint.kp, joint.kd) for joint in settings.joints], [(2.0, 0.05)] * 3)
        self.assertEqual((settings.control_rate_hz, settings.max_speed_deg_s), (200.0, 60.0))
        self.assertEqual(settings.can.interface, "socketcan")
        self.assertEqual(settings.can.socketcan_channel, "can_neck")
        self.assertIsNone(settings.can.slcan_port)
        self.assertEqual(settings.can.bitrate, 1_000_000)
        self.assertEqual(settings.can.adapter_usb_id, (0x16D0, 0x117E))

    def test_resolve_passes_loaded_settings_through(self) -> None:
        settings = load_config()
        self.assertIs(resolve_config(settings), settings)


class InvalidFileTest(unittest.TestCase):
    def load(self, data: Any) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neck.yaml"
            path.write_text(data if isinstance(data, str) else yaml.safe_dump(data))
            load_config(path)

    def edited(self, edit) -> dict[str, Any]:
        data = copy.deepcopy(DEFAULT)
        edit(data)
        return data

    def test_names_the_file(self) -> None:
        with self.assertRaisesRegex(ConfigError, r"neck\.yaml: can: missing bitrate"):
            self.load(self.edited(lambda d: d["can"].pop("bitrate")))

    def test_a_misspelled_key_is_missing_not_ignored(self) -> None:
        def misspell(data: dict[str, Any]) -> None:
            data["joints"][0]["reverse"] = data["joints"][0].pop("reversed")

        with self.assertRaisesRegex(ConfigError, r"joints\[0\]: missing reversed"):
            self.load(self.edited(misspell))

    def test_rejects_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown max_torque"):
            self.load(self.edited(lambda d: d.update(max_torque=1)))
        with self.assertRaisesRegex(ConfigError, r"joints\[1\]: unknown gear_ratio"):
            self.load(self.edited(lambda d: d["joints"][1].update(gear_ratio=9)))

    def test_rejects_repeated_ids(self) -> None:
        with self.assertRaisesRegex(ConfigError, r"\[51\] repeat"):
            self.load(self.edited(lambda d: d["joints"][1].update(id=51)))

    def test_needs_each_joint_once(self) -> None:
        with self.assertRaisesRegex(ConfigError, "expected one of yaw, roll, pitch, not 'tilt'"):
            self.load(self.edited(lambda d: d["joints"][2].update(name="tilt")))
        with self.assertRaisesRegex(ConfigError, "each of yaw, roll, pitch once"):
            self.load(self.edited(lambda d: d["joints"].pop()))

    def test_checks_types_and_ranges(self) -> None:
        cases = {
            "kp: expected a number": lambda d: d["joints"][0].update(kp="0.13"),
            "reversed: expected true or false": lambda d: d["joints"][0].update(reversed=1),
            r"id: 300 is outside 1-252": lambda d: d["joints"][0].update(id=300),
            "min_deg must be below max_deg": lambda d: d["joints"][1].update(min_deg=40),
            "control_rate_hz: must be above 0": lambda d: d.update(control_rate_hz=0),
            "can.interface: expected one of socketcan, slcan": lambda d: d["can"].update(
                interface="pcan"
            ),
            r"can.adapter_usb_id: expected \[vendor id, product id\]": lambda d: d["can"].update(
                adapter_usb_id=[0x16D0]
            ),
        }
        for message, edit in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ConfigError, message):
                self.load(self.edited(edit))

    def test_rejects_invalid_yaml(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not valid YAML"):
            self.load("joints: [\n")

    def test_a_missing_file_says_where_it_looked(self) -> None:
        with self.assertRaisesRegex(ConfigError, "no settings file at .*absent.yaml$"):
            load_config("absent.yaml")

    def test_a_missing_default_file_points_at_config(self) -> None:
        with (
            mock.patch.object(loader, "DEFAULT_CONFIG_PATH", Path("/nonexistent/neck.yaml")),
            self.assertRaisesRegex(ConfigError, "pip install .*--config"),
        ):
            load_config()


if __name__ == "__main__":
    unittest.main()
