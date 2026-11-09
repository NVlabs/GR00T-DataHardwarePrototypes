# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Section and per-check gating in the data-check config.

`is_enabled` is opt-in (a missing section does not run) while `check_enabled`
is opt-out (a check the config never mentions does run). The asymmetry is
deliberate and load-bearing, so it is pinned here: hardware-specific YAMLs
enable only the sections their rig supports, but a switch added inside a
section must not retroactively disable that check for every config written
before the switch existed.

Run with: uv run pytest tests/data_processing/test_data_check_config_gates.py -v
"""

from real_scripts.data_processing.data_check import check_enabled, is_enabled


class TestIsEnabled:
    def test_missing_section_does_not_run(self):
        assert is_enabled({}, "gripper") is False

    def test_explicit_false_does_not_run(self):
        assert is_enabled({"gripper": {"enabled": False}}, "gripper") is False

    def test_section_without_the_key_does_not_run(self):
        assert is_enabled({"gripper": {"jump_threshold_mm": 50.0}}, "gripper") is False

    def test_explicit_true_runs(self):
        assert is_enabled({"gripper": {"enabled": True}}, "gripper") is True

    def test_a_mapping_runs_the_section(self):
        """The mapping says which checks are on, not whether the section is."""
        config = {"gripper": {"enabled": {"movement": False}}}
        assert is_enabled(config, "gripper") is True

    def test_a_mapping_that_names_nothing_still_runs(self):
        assert is_enabled({"gripper": {"enabled": {}}}, "gripper") is True


class TestCheckEnabled:
    def test_enabled_true_runs_every_named_check(self):
        section = {"enabled": True}
        assert check_enabled(section, "movement") is True
        assert check_enabled(section, "anything_at_all") is True

    def test_enabled_false_runs_none_of_them(self):
        assert check_enabled({"enabled": False}, "movement") is False

    def test_mapping_switches_off_by_name(self):
        section = {"enabled": {"movement": False, "jumps": True}}
        assert check_enabled(section, "movement") is False
        assert check_enabled(section, "jumps") is True

    def test_unnamed_check_defaults_on(self):
        """Opt-out: a config written before this switch existed keeps the check."""
        section = {"enabled": {"movement": False}}
        assert check_enabled(section, "range") is True

    def test_a_section_predating_the_key_keeps_its_checks(self):
        """`enabled: true` is what every snapshotted station config says."""
        section = {"enabled": True, "movement_threshold_mm": 10.0}
        assert check_enabled(section, "range") is True

    def test_default_is_overridable(self):
        assert check_enabled({"enabled": {}}, "movement", default=False) is False
