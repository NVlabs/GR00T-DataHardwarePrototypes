# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrations follow the glove (its board serial), not just the side file.

Run from data_collection_scripts with ``.venv/bin/python -m unittest discover -s tests``.
"""

import os
import shutil
import tempfile
import unittest

import yaml

from gui.dc_ui.core import calibration as cal
from gui.dc_ui.core import devices


def _config(serial, zero=0.0):
    return {
        "device_name": "dexumi_exoskeleton",
        "device_serial": serial,
        "calibration_method": "calibration_tool_v4",
        "directions": "+" * 15,
        "zero_position_angle": [zero] * 15,
        "num_hand_joints": 16,
        "encoder_sensor_ids": list(range(15)),
        "joint_limits": [[-180, 180]] * 15,
    }


class GloveLibraryTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, cal.EXO_CONFIG_DIR))
        self._saved = (cal.repo_path, devices.repo_path)
        join = lambda rel: os.path.join(self.root, rel)  # noqa: E731
        cal.repo_path = devices.repo_path = join

    def tearDown(self):
        cal.repo_path, devices.repo_path = self._saved
        shutil.rmtree(self.root)

    def write(self, rel, config):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(config, f)

    def active(self, side):
        return cal.glove_status(side)["serial"]

    def test_a_connected_glove_without_a_calibration_is_not_calibrated(self):
        self.write(cal.config_rel("Left"), _config("OLDL"))
        (board,) = cal.boards_report(["NEWL"])
        self.assertFalse(board["calibrated"])
        self.assertIsNone(board["recorded_by"])

    def test_the_connected_glove_with_a_calibration_is_calibrated(self):
        self.write(cal.config_rel("Left"), _config("OLDL"))
        (board,) = cal.boards_report(["OLDL"])
        self.assertTrue(board["calibrated"])
        self.assertEqual(board["recorded_by"], "DEXUMI-Left")

    def test_archive_keeps_the_side_calibration_under_its_serial(self):
        self.write(cal.config_rel("Left"), _config("OLDL"))
        self.assertEqual(cal.archive("Left"), "OLDL")
        (entry,) = cal.library_entries()
        self.assertEqual(
            (entry["side"], entry["serial"], entry["calibrated"]),
            ("Left", "OLDL", True),
        )

    def test_a_saved_glove_is_loaded_when_plugged_in_and_the_other_is_kept(self):
        self.write(cal.config_rel("Left"), _config("NEWL", zero=1.0))
        self.write(cal.library_rel("Left", "OLDL"), _config("OLDL", zero=2.0))
        loaded = cal.activate_saved(["OLDL"])
        self.assertEqual(loaded, [{"side": "Left", "serial": "OLDL"}])
        self.assertEqual(self.active("Left"), "OLDL")
        serials = {e["serial"] for e in cal.library_entries()}
        self.assertEqual(serials, {"OLDL", "NEWL"})
        cal.activate_saved(["NEWL"])
        self.assertEqual(self.active("Left"), "NEWL")

    def test_loading_is_idempotent(self):
        self.write(cal.config_rel("Left"), _config("A"))
        self.assertEqual(cal.activate_saved(["A"]), [])
        self.assertEqual(cal.activate_saved(["A"]), [])

    def test_an_unknown_glove_is_not_guessed(self):
        self.write(cal.config_rel("Left"), _config("A"))
        self.assertEqual(cal.activate_saved(["STRANGER"]), [])
        self.assertEqual(self.active("Left"), "A")

    def test_a_glove_in_use_is_never_displaced(self):
        self.write(cal.config_rel("Left"), _config("A"))
        self.write(cal.library_rel("Left", "B"), _config("B"))
        self.assertEqual(cal.activate_saved(["A", "B"]), [])
        self.assertEqual(self.active("Left"), "A")

    def test_two_plugged_in_gloves_wanting_one_side_are_not_guessed(self):
        self.write(cal.library_rel("Left", "B"), _config("B"))
        self.write(cal.library_rel("Left", "C"), _config("C"))
        self.assertEqual(cal.activate_saved(["B", "C"]), [])

    def test_a_pair_loads_both_sides(self):
        self.write(cal.config_rel("Left"), _config("XL"))
        self.write(cal.config_rel("Right"), _config("XR"))
        self.write(cal.library_rel("Left", "PL"), _config("PL"))
        self.write(cal.library_rel("Right", "PR"), _config("PR"))
        loaded = cal.activate_saved(["PL", "PR"])
        self.assertEqual(
            {(g["side"], g["serial"]) for g in loaded},
            {("Left", "PL"), ("Right", "PR")},
        )
        self.assertEqual(
            [b["calibrated"] for b in cal.boards_report(["PL", "PR"])], [True, True]
        )
        self.assertTrue({"XL", "XR"} <= {e["serial"] for e in cal.library_entries()})

    def test_a_library_file_that_does_not_match_its_name_is_not_trusted(self):
        self.write(cal.library_rel("Left", "B"), _config("SOMEONEELSE"))
        self.assertEqual(cal.activate_saved(["B"]), [])

    def test_saved_gloves_lists_only_gloves_that_are_not_plugged_in(self):
        self.write(cal.config_rel("Left"), _config("A"))
        self.write(cal.library_rel("Right", "Z"), _config("Z"))
        saved = cal.saved_gloves(["A"])
        self.assertEqual([(g["side"], g["serial"]) for g in saved], [("Right", "Z")])


if __name__ == "__main__":
    unittest.main()
