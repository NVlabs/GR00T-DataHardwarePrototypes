# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless rendering regressions for non-obstructing recording warnings.

Run from data_collection_scripts with ``.venv/bin/python -m unittest discover -s tests``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from umi.common.utility import collection_dashboard as dashboard
from umi.common.utility.data_visualizer import DataVisualizer, VisualizationConfig


WARNINGS = {
    ("left_tracking_camera", "frame_drop"): "6 consecutive drops",
    ("right_umi_camera", "frame_delay"): "Latest frame 180 ms late",
}


def hand_data():
    data = {}
    for side in ("left", "right"):
        values = {
            f"tactile_{sensor}_{axis}_{point}": 12.5
            for sensor in range(33, 38)
            for axis in ("fx", "fy", "fz")
            for point in (1, 2)
        }
        data[f"{side}_dexumi_exoskeleton_tactile"] = [SimpleNamespace(**values)]
        data[f"{side}_dexumi_exoskeleton_encoder"] = [
            SimpleNamespace(joint_angles=np.linspace(1, 30, 15))
        ]
    return data


class WarningStripTests(unittest.TestCase):
    def setUp(self):
        self.camera = np.full((800, 1920, 3), (65, 55, 45), np.uint8)
        self.numeric = hand_data()

    def render(self, **kwargs):
        options = dict(
            size=(1600, 900),
            is_recording=True,
            saved_data_hours=1.25,
            task_description="Pick and place",
            start_hint="W  Stop and save",
        )
        options.update(kwargs)
        return dashboard.render_dashboard(self.camera, self.numeric, **options)

    def test_warnings_only_change_their_reserved_card_at_all_window_sizes(self):
        for size in ((960, 600), (1600, 900), (1800, 1100), (2400, 900)):
            with self.subTest(size=size):
                clear_info, active_info = {}, {}
                clear = self.render(size=size, layout_info=clear_info)
                active = self.render(
                    size=size, warnings=WARNINGS, layout_info=active_info
                )
                self.assertEqual(clear.shape, (size[1], size[0], 3))
                self.assertEqual(clear_info, active_info)
                x, y, w, h = clear_info["warning_box"]
                changed = np.any(active != clear, axis=2)
                self.assertTrue(changed[y : y + h + 1, x : x + w + 1].any())
                changed[y : y + h + 1, x : x + w + 1] = False
                # Includes the complete header, camera, tactile/encoder tables,
                # and footer: warnings must not redraw or move any of them.
                self.assertFalse(changed.any())
        np.testing.assert_array_equal(
            self.camera, np.full_like(self.camera, (65, 55, 45))
        )

    def test_many_long_warnings_stay_bounded_and_every_warning_gets_a_page(self):
        warnings = {
            (f"source_{index:02d}", "frame_delay"): "Delayed frame " * 100
            for index in range(9)
        }
        info = {}
        clear = self.render(size=(960, 600), layout_info=info)
        x, y, w, h = info["warning_box"]
        seen = set()
        for page in range(5):
            with patch.object(dashboard, "_text", wraps=dashboard._text) as text:
                active = self.render(
                    size=(960, 600), warnings=warnings, warning_page=page
                )
            labels = [str(call.args[1]) for call in text.call_args_list]
            sources = {label for label in labels if label.startswith("Source ")}
            self.assertLessEqual(len(sources), 2)
            seen.update(sources)
            self.assertIn("WARNINGS  9 active", labels)
            self.assertTrue(any("of 9 (cycles)" in label for label in labels))
            changed = np.any(active != clear, axis=2)
            changed[y : y + h + 1, x : x + w + 1] = False
            self.assertFalse(changed.any())
        self.assertEqual(seen, {f"Source {i:02d} / Frame delay" for i in range(9)})
        np.testing.assert_array_equal(
            self.render(warnings=warnings, warning_page=0),
            self.render(warnings=warnings, warning_page=5),
        )

    def test_all_sensor_rows_and_recording_controls_still_fit(self):
        for components in (("fz",), ("fx", "fy", "fz")):
            with self.subTest(components=components):
                info = {}
                with patch.object(dashboard, "_text", wraps=dashboard._text) as text:
                    self.render(
                        size=(960, 600),
                        components=components,
                        warnings=WARNINGS,
                        layout_info=info,
                    )
                labels = [str(call.args[1]) for call in text.call_args_list]
                for joint in range(15):
                    self.assertEqual(labels.count(f"J{joint:02d}"), 2)
                for finger in (
                    "Thumb 33",
                    "Index 34",
                    "Middle 35",
                    "Ring 36",
                    "Pinky 37",
                ):
                    self.assertEqual(labels.count(finger), 2)
                self.assertEqual(labels.count("12.5"), 20 * len(components))
                self.assertIn("W  Stop and save", labels)
                self.assertIn("A Discard", labels)
                self.assertIn("Q Quit", labels)
                warning_y = info["warning_box"][1]
                for call in text.call_args_list:
                    if str(call.args[1]).startswith("J") or str(call.args[1]) == "12.5":
                        self.assertLess(call.args[3], warning_y)

    def visualizer(self, size=(1600, 900)):
        return DataVisualizer(
            VisualizationConfig(
                dashboard=True,
                window_width=size[0],
                window_height=size[1],
                tactile_components=("fz",),
            )
        )

    def test_pages_have_five_seconds_even_when_warning_messages_update(self):
        warnings = {
            (f"source_{index:02d}", "frame_delay"): "Frame delayed"
            for index in range(5)
        }
        viz = self.visualizer()
        pages = []
        for now in (100.0, 104.9, 105.0, 110.0):
            warnings[("source_00", "frame_delay")] = f"Delay measured at {now}"
            with patch(
                "umi.common.utility.data_visualizer.time.monotonic", return_value=now
            ):
                with patch.object(dashboard, "_text", wraps=dashboard._text) as text:
                    viz.draw_dashboard(self.camera, 42, True, warnings=warnings)
            pages.append(
                {
                    call.args[1]
                    for call in text.call_args_list
                    if "of 5 (cycles)" in str(call.args[1])
                }
            )
        self.assertEqual(
            pages,
            [
                {"1-2 of 5 (cycles)"},
                {"1-2 of 5 (cycles)"},
                {"3-4 of 5 (cycles)"},
                {"5-5 of 5 (cycles)"},
            ],
        )

    def test_fluctuating_sources_do_not_starve_later_pages_and_clearing_resets(self):
        warnings = {
            (f"source_{index:02d}", "frame_delay"): "Frame delayed"
            for index in range(5)
        }
        viz = self.visualizer()
        for now, active, expected in (
            (100, warnings, "1-2 of 5 (cycles)"),
            (
                106,
                warnings | {("source_99", "frame_drop"): "Frame dropped"},
                "3-4 of 6 (cycles)",
            ),
            (107, {}, "No active warnings"),
            (108, warnings, "1-2 of 5 (cycles)"),
        ):
            with self.subTest(now=now):
                with patch(
                    "umi.common.utility.data_visualizer.time.monotonic",
                    return_value=now,
                ):
                    with patch.object(
                        dashboard, "_text", wraps=dashboard._text
                    ) as text:
                        viz.draw_dashboard(self.camera, 42, True, warnings=active)
                self.assertIn(expected, [call.args[1] for call in text.call_args_list])

    def test_warning_status_matches_the_recording_state(self):
        for recording, error, expected in (
            (True, False, "Recording continues"),
            (False, False, "Not recording"),
            (False, True, "Resolve error to continue"),
        ):
            with self.subTest(recording=recording, error=error):
                viz = self.visualizer()
                with patch.object(dashboard, "_text", wraps=dashboard._text) as text:
                    viz.draw_dashboard(
                        self.camera,
                        42,
                        recording,
                        error_active=error,
                        warnings={("right_umi_camera", "frame_drop"): "Dropped."},
                    )
                labels = [str(call.args[1]) for call in text.call_args_list]
                self.assertIn(expected, labels)
                self.assertIn("Right UMI (Wrist) / Frame drop", labels)


if __name__ == "__main__":
    unittest.main()
