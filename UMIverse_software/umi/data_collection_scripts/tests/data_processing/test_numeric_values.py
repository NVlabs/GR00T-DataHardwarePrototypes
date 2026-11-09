# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generic numeric_values check.

Run with: uv run pytest tests/data_processing/test_numeric_values.py -v
"""

import numpy as np
import pytest
import zarr

from real_scripts.data_processing.data_check import (
    check_numeric_array,
    check_numeric_values,
    run_numeric_value_checks,
)
from real_scripts.data_processing.data_check.numeric import (
    _resolve_value,
    _walk_recording_config_path,
)

# ---------------------------------------------------------------------------
# check_numeric_array — pure-numpy core
# ---------------------------------------------------------------------------


class TestExistsHandling:
    """`exists` is handled at the zarr layer (check_numeric_values), not in
    check_numeric_array itself. check_numeric_array works against an already-
    loaded array, so it has nothing to do with existence."""

    def test_check_numeric_array_no_exists_subcheck(self):
        result = check_numeric_array(np.array([1.0, 2.0, 3.0]), {"no_nan": True})
        assert "exists" not in result


class TestNoNan:
    def test_passes_when_clean(self):
        result = check_numeric_array(np.array([1.0, 2.0, 3.0]), {"no_nan": True})
        assert result["no_nan"]["passed"] is True
        assert result["no_nan"]["num_nans"] == 0
        assert result["no_nan"]["nan_positions"] == []

    def test_fails_with_nan_in_1d(self):
        values = np.array([1.0, np.nan, 3.0])
        result = check_numeric_array(values, {"no_nan": True})
        assert result["no_nan"]["passed"] is False
        assert result["no_nan"]["num_nans"] == 1
        assert result["no_nan"]["nan_positions"][0]["frame_index"] == 1
        assert result["no_nan"]["nan_positions"][0]["axis_index"] is None

    def test_fails_with_nan_in_2d_reports_axis(self):
        values = np.array([[0.1, 0.2, 0.3], [0.4, np.nan, 0.6], [0.7, 0.8, 0.9]])
        result = check_numeric_array(values, {"no_nan": True})
        assert result["no_nan"]["passed"] is False
        assert result["no_nan"]["num_nans"] == 1
        pos = result["no_nan"]["nan_positions"][0]
        assert pos["frame_index"] == 1
        assert pos["axis_index"] == 1


class TestInRange:
    def test_passes_when_all_in_range(self):
        result = check_numeric_array(
            np.array([0.0, 0.5, 1.0]), {"in_range": [0.0, 1.0]}
        )
        assert result["in_range"]["passed"] is True
        assert result["in_range"]["num_out_of_range"] == 0
        assert result["in_range"]["violations"] == []
        assert result["in_range"]["actual_min"] == 0.0
        assert result["in_range"]["actual_max"] == 1.0

    def test_fails_with_below_and_above(self):
        values = np.array([-1.0, 0.5, 2.0])
        result = check_numeric_array(values, {"in_range": [0.0, 1.0]})
        assert result["in_range"]["passed"] is False
        assert result["in_range"]["num_out_of_range"] == 2
        frame_indices = [v["frame_index"] for v in result["in_range"]["violations"]]
        assert frame_indices == [0, 2]
        assert result["in_range"]["violations"][0]["value"] == -1.0
        assert result["in_range"]["violations"][1]["value"] == 2.0

    def test_2d_reports_axis_index(self):
        values = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 99.0, 0.6],  # element [1, 1] out of range
                [0.7, 0.8, 0.9],
            ]
        )
        result = check_numeric_array(values, {"in_range": [0.0, 1.0]})
        assert result["in_range"]["passed"] is False
        assert result["in_range"]["num_out_of_range"] == 1
        v = result["in_range"]["violations"][0]
        assert v["frame_index"] == 1
        assert v["axis_index"] == 1
        assert v["value"] == 99.0

    def test_rejects_malformed_spec(self):
        with pytest.raises(ValueError):
            check_numeric_array(np.array([1.0]), {"in_range": [0.0]})


class TestJump:
    def test_passes_below_threshold(self):
        result = check_numeric_array(np.array([1.0, 1.1, 1.2]), {"jump": 0.5})
        assert result["jump"]["passed"] is True
        assert result["jump"]["num_jumps"] == 0

    def test_fails_at_threshold_crossing(self):
        values = np.array([1.0, 1.1, 5.0, 5.1])  # jump 3.9 between idx 1 and 2
        result = check_numeric_array(values, {"jump": 0.5})
        assert result["jump"]["passed"] is False
        assert result["jump"]["num_jumps"] == 1
        j = result["jump"]["jump_frames"][0]
        assert j["frame_index"] == 2  # the frame that jumped to
        assert j["before_value"] == 1.1
        assert j["after_value"] == 5.0
        assert j["magnitude"] == pytest.approx(3.9, abs=1e-9)

    def test_2d_per_axis_jump(self):
        # axis 1 jumps from 0 -> 10 at frame 1; axis 0 stays smooth
        values = np.array([[1.0, 0.0], [1.1, 10.0], [1.2, 10.1]])
        result = check_numeric_array(values, {"jump": 0.5})
        assert result["jump"]["passed"] is False
        assert result["jump"]["num_jumps"] == 1
        j = result["jump"]["jump_frames"][0]
        assert j["frame_index"] == 1
        assert j["axis_index"] == 1

    def test_too_short_for_diff_passes(self):
        result = check_numeric_array(np.array([1.0]), {"jump": 0.5})
        assert result["jump"]["passed"] is True
        assert result["jump"]["num_jumps"] == 0


class TestMovement:
    def test_passes_with_sufficient_range(self):
        result = check_numeric_array(np.array([0.0, 0.5, 1.0]), {"movement": 0.5})
        assert result["movement"]["passed"] is True
        assert result["movement"]["max_range"] == 1.0

    def test_fails_when_below_min_range(self):
        result = check_numeric_array(np.array([0.0, 0.01, 0.02]), {"movement": 0.5})
        assert result["movement"]["passed"] is False

    def test_2d_passes_if_any_axis_moves(self):
        # axis 0 doesn't move, axis 1 moves a lot
        values = np.array([[1.0, 0.0], [1.0, 5.0], [1.0, 10.0]])
        result = check_numeric_array(values, {"movement": 0.5})
        assert result["movement"]["passed"] is True
        assert result["movement"]["per_axis_range"] == [0.0, 10.0]


class TestMovementRequiredAxes:
    """Dict-form movement spec: {threshold, required_axes} — each listed axis
    must span >= threshold, unlisted axes are unconstrained."""

    SPEC = {"movement": {"threshold": 5.0, "required_axes": [0, 1, 3]}}

    def test_passes_when_all_required_axes_move(self):
        # axes 0, 1, 3 each span >= 5; axis 2 stays still (unlisted, ignored)
        values = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [6.0, 5.0, 1.0, 7.0],
            ]
        )
        result = check_numeric_array(values, self.SPEC)
        assert result["movement"]["passed"] is True
        assert result["movement"]["required_axes"] == [0, 1, 3]
        assert result["movement"]["failing_axes"] == []
        # Existing keys retained for report consumers.
        assert result["movement"]["min_range"] == 5.0
        assert result["movement"]["per_axis_range"] == [6.0, 5.0, 0.0, 7.0]
        assert result["movement"]["max_range"] == 7.0

    def test_fails_when_one_required_axis_still(self):
        # axis 3 only spans 0.4 — must fail even though axes 0, 1 move plenty
        values = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 10.0, 0.0, 0.4],
            ]
        )
        result = check_numeric_array(values, self.SPEC)
        assert result["movement"]["passed"] is False
        failing = result["movement"]["failing_axes"]
        assert len(failing) == 1
        assert failing[0]["axis_index"] == 3
        assert failing[0]["range"] == pytest.approx(0.4)

    def test_unlisted_axis_movement_does_not_rescue(self):
        # An any-axis rule would pass on axis 2's movement; the dict form
        # must not — required axes 0, 1, 3 are all still.
        values = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 50.0, 0.0],
            ]
        )
        result = check_numeric_array(values, self.SPEC)
        assert result["movement"]["passed"] is False
        assert [f["axis_index"] for f in result["movement"]["failing_axes"]] == [
            0,
            1,
            3,
        ]

    def test_1d_stream_axis_zero(self):
        # A scalar stream has one flat axis (index 0).
        values = np.array([0.0, 1.0, 6.0])
        result = check_numeric_array(
            values, {"movement": {"threshold": 5.0, "required_axes": [0]}}
        )
        assert result["movement"]["passed"] is True

    def test_multi_d_uses_flat_axis_indexing(self):
        # Shape (frames, 2, 2) — flat axis order matches per_axis_range: axes
        # are raveled row-major, so flat index 3 = [1][1].
        values = np.zeros((3, 2, 2))
        values[:, 1, 1] = [0.0, 3.0, 6.0]
        result = check_numeric_array(
            values, {"movement": {"threshold": 5.0, "required_axes": [3]}}
        )
        assert result["movement"]["passed"] is True
        result = check_numeric_array(
            values, {"movement": {"threshold": 5.0, "required_axes": [0]}}
        )
        assert result["movement"]["passed"] is False

    def test_out_of_bounds_axis_fails_gracefully(self):
        values = np.zeros((4, 15))
        result = check_numeric_array(
            values, {"movement": {"threshold": 5.0, "required_axes": [20]}}
        )
        assert result["movement"]["passed"] is False
        assert result["movement"]["out_of_bounds_axes"] == [20]

    def test_rejects_missing_threshold(self):
        with pytest.raises(ValueError):
            check_numeric_array(np.zeros(3), {"movement": {"required_axes": [0]}})

    def test_rejects_empty_required_axes(self):
        with pytest.raises(ValueError):
            check_numeric_array(
                np.zeros(3), {"movement": {"threshold": 5.0, "required_axes": []}}
            )

    def test_rejects_negative_axis(self):
        with pytest.raises(ValueError):
            check_numeric_array(
                np.zeros(3), {"movement": {"threshold": 5.0, "required_axes": [-1]}}
            )

    def test_rejects_non_int_axis(self):
        with pytest.raises(ValueError):
            check_numeric_array(
                np.zeros(3),
                {"movement": {"threshold": 5.0, "required_axes": [0, "1"]}},
            )


class TestMovementPerAxisThresholds:
    """Dict-form movement spec with `required_axes` as an {axis: threshold}
    mapping — each listed axis must span its own threshold."""

    SPEC = {"movement": {"required_axes": {0: 5.0, 1: 8.0, 3: 2.0}}}

    def test_passes_when_each_axis_meets_own_threshold(self):
        values = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [6.0, 9.0, 1.0, 3.0],
            ]
        )
        result = check_numeric_array(values, self.SPEC)
        assert result["movement"]["passed"] is True
        assert result["movement"]["required_axes"] == [0, 1, 3]
        assert result["movement"]["axis_thresholds"] == {0: 5.0, 1: 8.0, 3: 2.0}
        assert result["movement"]["failing_axes"] == []
        # No single shared threshold in this form.
        assert "min_range" not in result["movement"]

    def test_fails_only_the_axis_below_its_own_threshold(self):
        # axis 0 spans 6 (>= 5, pass); axis 1 spans 6 too but its threshold is
        # 8 — a uniform threshold could not express this outcome.
        values = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [6.0, 6.0, 0.0, 3.0],
            ]
        )
        result = check_numeric_array(values, self.SPEC)
        assert result["movement"]["passed"] is False
        failing = result["movement"]["failing_axes"]
        assert len(failing) == 1
        assert failing[0]["axis_index"] == 1
        assert failing[0]["range"] == pytest.approx(6.0)
        assert failing[0]["threshold"] == 8.0

    def test_unlisted_axis_movement_does_not_rescue(self):
        values = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 50.0, 0.0],
            ]
        )
        result = check_numeric_array(values, self.SPEC)
        assert result["movement"]["passed"] is False
        assert [f["axis_index"] for f in result["movement"]["failing_axes"]] == [
            0,
            1,
            3,
        ]

    def test_out_of_bounds_axis_fails_gracefully(self):
        values = np.zeros((4, 15))
        result = check_numeric_array(values, {"movement": {"required_axes": {20: 5.0}}})
        assert result["movement"]["passed"] is False
        assert result["movement"]["out_of_bounds_axes"] == [20]

    def test_uniform_form_also_reports_axis_thresholds(self):
        # The list form is normalized to per-axis pairs internally; its report
        # keeps min_range and gains axis_thresholds + per-entry threshold.
        values = np.zeros((2, 4))
        result = check_numeric_array(
            values, {"movement": {"threshold": 5.0, "required_axes": [0, 3]}}
        )
        assert result["movement"]["min_range"] == 5.0
        assert result["movement"]["axis_thresholds"] == {0: 5.0, 3: 5.0}
        assert result["movement"]["failing_axes"][0]["threshold"] == 5.0

    def test_rejects_mapping_combined_with_threshold(self):
        with pytest.raises(ValueError):
            check_numeric_array(
                np.zeros(3),
                {"movement": {"threshold": 5.0, "required_axes": {0: 5.0}}},
            )

    def test_rejects_empty_mapping(self):
        with pytest.raises(ValueError):
            check_numeric_array(np.zeros(3), {"movement": {"required_axes": {}}})

    def test_rejects_negative_axis_key(self):
        with pytest.raises(ValueError):
            check_numeric_array(np.zeros(3), {"movement": {"required_axes": {-1: 5.0}}})

    def test_accepts_digit_string_axis_keys(self):
        # Specs delivered through JSON (for example task-registry
        # data_check_overrides) can only carry string keys — {0: 5.0} arrives as {"0": 5.0}.
        values = np.array([[0.0, 0.0], [6.0, 1.0]])
        result = check_numeric_array(
            values, {"movement": {"required_axes": {"0": 5.0}}}
        )
        assert result["movement"]["passed"] is True
        assert result["movement"]["required_axes"] == [0]
        assert result["movement"]["axis_thresholds"] == {0: 5.0}

    def test_rejects_non_digit_string_axis_key(self):
        with pytest.raises(ValueError):
            check_numeric_array(
                np.zeros(3), {"movement": {"required_axes": {"a": 5.0}}}
            )

    def test_rejects_negative_string_axis_key(self):
        with pytest.raises(ValueError):
            check_numeric_array(
                np.zeros(3), {"movement": {"required_axes": {"-1": 5.0}}}
            )

    def test_rejects_non_numeric_threshold_value(self):
        with pytest.raises(ValueError):
            check_numeric_array(np.zeros(3), {"movement": {"required_axes": {0: "5"}}})


class TestShape:
    def test_2d_matches_expected(self):
        result = check_numeric_array(np.zeros((10, 15)), {"shape": [15]})
        assert result["shape"]["passed"] is True
        assert result["shape"]["expected"] == [15]
        assert result["shape"]["actual"] == [15]

    def test_2d_mismatch(self):
        result = check_numeric_array(np.zeros((10, 12)), {"shape": [15]})
        assert result["shape"]["passed"] is False
        assert result["shape"]["actual"] == [12]

    def test_1d_scalar_stream(self):
        result = check_numeric_array(np.zeros(10), {"shape": []})
        assert result["shape"]["passed"] is True
        assert result["shape"]["expected"] == []
        assert result["shape"]["actual"] == []


class TestStartWithinTolerance:
    """`start_within_tolerance` pins frame 0 to a target ± tolerance."""

    def test_passes_when_first_frame_within_tolerance(self):
        # 15-D vector — close to target on every axis.
        target = np.arange(15, dtype=float)
        first = target + np.array([0.3, -0.4, 0.1] * 5)  # all within 0.5
        rest = np.zeros((5, 15))  # any subsequent values
        values = np.vstack([first, rest])
        result = check_numeric_array(
            values,
            {"start_within_tolerance": {"target": target.tolist(), "tolerance": 0.5}},
        )
        swt = result["start_within_tolerance"]
        assert swt["passed"] is True
        assert swt["n_in_tolerance"] == 15
        assert swt["total_axes"] == 15

    def test_fails_and_reports_worst_axis(self):
        target = np.zeros(15)
        first = np.zeros(15)
        first[7] = 10.0  # one axis way off
        values = np.vstack([first, np.zeros((3, 15))])
        result = check_numeric_array(
            values,
            {"start_within_tolerance": {"target": target.tolist(), "tolerance": 5.0}},
        )
        swt = result["start_within_tolerance"]
        assert swt["passed"] is False
        assert swt["worst_axis_index"] == 7
        assert swt["worst_deviation_signed"] == pytest.approx(10.0)
        assert swt["n_in_tolerance"] == 14

    def test_shape_mismatch_yields_error(self):
        values = np.zeros((3, 10))  # 10-D, but target is 15-D
        result = check_numeric_array(
            values,
            {"start_within_tolerance": {"target": [0.0] * 15, "tolerance": 1.0}},
        )
        swt = result["start_within_tolerance"]
        assert swt["passed"] is False
        assert "shape mismatch" in swt["error"]

    def test_scalar_stream_first_value_zero(self):
        """1-D scalar stream (shape (N,)) with scalar target.
        Pins the first value to 0.0 exactly; values[0] is a 0-D numpy scalar."""
        values = np.array([0.0, 0.0, 1.5, 3.2])
        result = check_numeric_array(
            values,
            {"start_within_tolerance": {"target": 0.0, "tolerance": 0.0}},
        )
        swt = result["start_within_tolerance"]
        assert swt["passed"] is True
        assert swt["actual_first_frame"] == [0.0]
        assert swt["abs_deviations"] == [0.0]
        assert swt["worst_axis_index"] == 0

    def test_scalar_stream_nonzero_first_fails(self):
        values = np.array([2.5, 0.0, 0.0])
        result = check_numeric_array(
            values,
            {"start_within_tolerance": {"target": 0.0, "tolerance": 0.0}},
        )
        swt = result["start_within_tolerance"]
        assert swt["passed"] is False
        assert swt["worst_deviation_signed"] == pytest.approx(2.5)


class TestRecordingConfigPath:
    """`_walk_recording_config_path` + `_resolve_value` — the YAML escape hatch
    for pulling targets/tolerances out of the recording config so the data
    check uses the same source of truth the pre-episode check does."""

    cfg = {
        "recording": {"record_fps": 45},
        "pre_episode_checks": [
            {"name": "umi_device.uart_connected"},
            {
                "name": "example.start_target",
                "params": {
                    "tolerance_deg": 5,
                    "target_reading": [1.0, 2.0, 3.0],
                },
            },
        ],
    }

    def test_walk_dict_keys(self):
        assert _walk_recording_config_path(self.cfg, ["recording", "record_fps"]) == 45

    def test_walk_list_filter_by_name(self):
        out = _walk_recording_config_path(
            self.cfg,
            [
                "pre_episode_checks",
                {"name": "example.start_target"},
                "params",
                "tolerance_deg",
            ],
        )
        assert out == 5

    def test_walk_unknown_key_raises(self):
        with pytest.raises(ValueError, match="not found"):
            _walk_recording_config_path(self.cfg, ["recording", "nonexistent"])

    def test_walk_no_matching_list_filter_raises(self):
        with pytest.raises(ValueError, match="no list item matched"):
            _walk_recording_config_path(
                self.cfg,
                ["pre_episode_checks", {"name": "does_not_exist"}],
            )

    def test_resolve_passthrough_when_not_a_lookup(self):
        assert _resolve_value(42, self.cfg) == 42
        assert _resolve_value([1, 2, 3], self.cfg) == [1, 2, 3]

    def test_resolve_pulls_from_recording_config(self):
        spec = {
            "from_recording_config": [
                "pre_episode_checks",
                {"name": "example.start_target"},
                "params",
                "target_reading",
            ]
        }
        assert _resolve_value(spec, self.cfg) == [1.0, 2.0, 3.0]

    def test_resolve_without_recording_config_raises(self):
        spec = {"from_recording_config": ["recording", "record_fps"]}
        with pytest.raises(ValueError, match="recording config"):
            _resolve_value(spec, None)


# ---------------------------------------------------------------------------
# check_numeric_values — zarr layer (exists handling + result aggregation)
# ---------------------------------------------------------------------------


def _build_episode(tmp_path, arrays):
    """Create a zarr episode with the given {device: {key: ndarray}} mapping."""
    store = zarr.open(str(tmp_path / "episode_0"), mode="w")
    for device, keys in arrays.items():
        grp = store.create_group(device)
        for k, arr in keys.items():
            zarr_arr = grp.create_array(k, shape=arr.shape, dtype=arr.dtype)
            zarr_arr[:] = arr
    return store


class TestExistsAtZarrLayer:
    def test_passes_when_array_present(self, tmp_path):
        # Non-zero data on purpose: zarr v3 elides chunks that equal the
        # fill_value (default 0.0 for float64), so an all-zeros fixture would
        # trip the "no chunks written" check even though it was written to.
        ep = _build_episode(
            tmp_path,
            {"right_encoder": {"joint_angles": np.ones((10, 15))}},
        )
        result = check_numeric_values(
            ep,
            "right_encoder",
            "joint_angles",
            {"exists": True, "shape": [15]},
        )
        assert result["exists"]["key_present"] is True
        assert result["passed"] is True

    def test_fails_when_array_missing(self, tmp_path):
        ep = _build_episode(
            tmp_path,
            {"right_encoder": {"raw_encoder_angles": np.zeros(10)}},
        )
        result = check_numeric_values(
            ep,
            "right_encoder",
            "joint_angles",
            {"exists": True, "no_nan": True},
        )
        assert result["exists"]["key_present"] is False
        assert result["passed"] is False
        assert "right_encoder.joint_angles: missing" in result["issues"]
        # When the array is missing, downstream sub-checks are skipped.
        assert result["no_nan"] is None

    def test_fails_when_array_registered_but_no_chunks(self, tmp_path):
        """An array can be registered (zarr.json present) without any chunks
        being written — reading silently returns the fill_value. The exists
        check must catch this so the data-missing bug doesn't pass downstream
        checks (no_nan, in_range, …) by accident."""
        store = zarr.open(str(tmp_path / "episode_0"), mode="w")
        grp = store.create_group("d")
        # Create the array but never write to it.
        grp.create_array("a", shape=(10,), dtype="float64")
        assert grp["a"].nchunks_initialized == 0
        result = check_numeric_values(
            store, "d", "a", {"exists": True, "no_nan": True, "in_range": [-1.0, 1.0]}
        )
        assert result["passed"] is False
        assert result["exists"]["key_present"] is True
        assert result["exists"]["chunks_initialized"] == 0
        assert result["exists"]["total_chunks"] == 1
        assert any("no chunks written" in i for i in result["issues"])
        # Downstream sub-checks must be skipped — running them on fill_value
        # data would produce misleading "passed" results.
        assert result["no_nan"] is None
        assert result["in_range"] is None

    def test_silently_skips_when_no_exists_and_missing(self, tmp_path):
        ep = _build_episode(
            tmp_path,
            {"right_encoder": {"raw_encoder_angles": np.zeros(10)}},
        )
        result = check_numeric_values(
            ep,
            "right_encoder",
            "joint_angles",  # missing key, but exists: not set
            {"no_nan": True},
        )
        # No issue, no failure — silently skipped because user didn't ask.
        assert result["passed"] is True
        assert result["issues"] == []


class TestIssueWording:
    def test_emits_nan_issue(self, tmp_path):
        ep = _build_episode(
            tmp_path,
            {"d": {"a": np.array([1.0, np.nan, 3.0])}},
        )
        result = check_numeric_values(ep, "d", "a", {"no_nan": True})
        assert result["passed"] is False
        assert "d.a: 1 NaN values" in result["issues"]

    def test_emits_out_of_range_issue(self, tmp_path):
        ep = _build_episode(
            tmp_path,
            {"d": {"a": np.array([10.0, 20.0, 30.0])}},
        )
        result = check_numeric_values(ep, "d", "a", {"in_range": [0.0, 25.0]})
        assert "d.a: out of range" in result["issues"][0]
        assert "(1 elements)" in result["issues"][0]

    def test_emits_jump_issue(self, tmp_path):
        ep = _build_episode(
            tmp_path,
            {"d": {"a": np.array([1.0, 1.1, 5.0])}},
        )
        result = check_numeric_values(ep, "d", "a", {"jump": 0.5})
        assert "d.a: 1 jumps detected" in result["issues"][0]

    def test_emits_shape_mismatch_issue(self, tmp_path):
        ep = _build_episode(tmp_path, {"d": {"a": np.zeros((10, 7))}})
        result = check_numeric_values(ep, "d", "a", {"shape": [15]})
        assert "d.a: shape mismatch" in result["issues"][0]

    def test_emits_no_movement_issue(self, tmp_path):
        ep = _build_episode(tmp_path, {"d": {"a": np.zeros(10)}})
        result = check_numeric_values(ep, "d", "a", {"movement": 1.0})
        assert "d.a: no movement" in result["issues"][0]

    def test_emits_required_joint_issue_naming_failing_joints(self, tmp_path):
        # joints 0 and 1 move; joint 3 spans only 0.4 — issue must name joint 3
        values = np.zeros((2, 15))
        values[1, 0] = 10.0
        values[1, 1] = 10.0
        values[1, 3] = 0.4
        ep = _build_episode(tmp_path, {"d": {"a": values}})
        result = check_numeric_values(
            ep, "d", "a", {"movement": {"threshold": 5.0, "required_axes": [0, 1, 3]}}
        )
        assert result["passed"] is False
        issue = result["issues"][0]
        assert "d.a: required axes did not move" in issue
        assert "axis 3" in issue
        assert "axis 0" not in issue
        assert "axis 1" not in issue

    def test_per_axis_issue_names_each_axis_own_threshold(self, tmp_path):
        # joint 0 clears its 5-deg bar; joint 1 misses its 8-deg bar — the
        # issue must quote joint 1's own threshold, not joint 0's.
        values = np.zeros((2, 15))
        values[1, 0] = 6.0
        values[1, 1] = 6.0
        ep = _build_episode(tmp_path, {"d": {"a": values}})
        result = check_numeric_values(
            ep, "d", "a", {"movement": {"required_axes": {0: 5.0, 1: 8.0}}}
        )
        assert result["passed"] is False
        issue = result["issues"][0]
        assert "d.a: required axes did not move" in issue
        assert "axis 1: range 6.00 < 8.0" in issue
        assert "axis 0" not in issue


class TestStartWithinToleranceAtZarrLayer:
    """End-to-end: zarr fixture + `from_recording_config` resolves through the
    full check_numeric_values pipeline, the way a start target set by a
    pre-episode check is read back."""

    RECORDING_CONFIG = {
        "pre_episode_checks": [
            {
                "name": "example.start_target",
                "params": {
                    "tolerance_deg": 5.0,
                    "target_reading": [0.0, 1.0, 2.0],
                },
            },
        ],
    }

    SPEC = {
        "start_within_tolerance": {
            "target": {
                "from_recording_config": [
                    "pre_episode_checks",
                    {"name": "example.start_target"},
                    "params",
                    "target_reading",
                ]
            },
            "tolerance": {
                "from_recording_config": [
                    "pre_episode_checks",
                    {"name": "example.start_target"},
                    "params",
                    "tolerance_deg",
                ]
            },
        },
    }

    def test_passes_when_first_frame_in_tolerance(self, tmp_path):
        # First frame = target ± a bit (within 5°)
        values = np.array(
            [
                [0.5, 1.2, 2.3],  # frame 0: within tolerance
                [10.0, 10.0, 10.0],  # later frames don't matter
                [20.0, 20.0, 20.0],
            ],
            dtype=np.float64,
        )
        ep = _build_episode(tmp_path, {"d": {"joint_angles": values}})
        result = check_numeric_values(
            ep,
            "d",
            "joint_angles",
            self.SPEC,
            recording_config=self.RECORDING_CONFIG,
        )
        assert result["passed"] is True
        swt = result["start_within_tolerance"]
        assert swt["tolerance"] == 5.0
        assert swt["target"] == [0.0, 1.0, 2.0]
        assert swt["n_in_tolerance"] == 3

    def test_fails_when_first_frame_out_of_tolerance(self, tmp_path):
        values = np.array(
            [
                [0.0, 1.0, 99.0],  # axis 2 is 97° off from target 2.0
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        ep = _build_episode(tmp_path, {"d": {"joint_angles": values}})
        result = check_numeric_values(
            ep,
            "d",
            "joint_angles",
            self.SPEC,
            recording_config=self.RECORDING_CONFIG,
        )
        assert result["passed"] is False
        assert any("start frame out of tolerance" in i for i in result["issues"])
        swt = result["start_within_tolerance"]
        assert swt["worst_axis_index"] == 2

    def test_missing_recording_config_surfaces_a_clear_error(self, tmp_path):
        values = np.zeros((3, 3))
        ep = _build_episode(tmp_path, {"d": {"joint_angles": values + 1.0}})
        # Spec references recording config, but caller didn't pass one.
        result = check_numeric_values(
            ep, "d", "joint_angles", self.SPEC, recording_config=None
        )
        assert result["passed"] is False
        assert any("start_within_tolerance config error" in i for i in result["issues"])


# ---------------------------------------------------------------------------
# run_numeric_value_checks — dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_iterates_yaml_order_and_aggregates_issues(self, tmp_path):
        ep = _build_episode(
            tmp_path,
            {
                "deviceA": {"x": np.array([1.0, np.nan]), "y": np.zeros(3)},
                "deviceB": {"z": np.array([100.0])},
            },
        )
        config = {
            "enabled": True,
            "deviceA": {
                "x": {"no_nan": True},
                "y": {"movement": 0.5},
            },
            "deviceB": {
                "z": {"in_range": [0.0, 50.0]},
            },
        }
        results, issues = run_numeric_value_checks(ep, config)
        # Iteration is YAML order: deviceA.x, deviceA.y, deviceB.z.
        assert list(results.keys()) == ["deviceA", "deviceB"]
        assert list(results["deviceA"].keys()) == ["x", "y"]
        # Each failure emits at least one issue, in order.
        assert any("deviceA.x: 1 NaN values" in i for i in issues)
        assert any("deviceA.y: no movement" in i for i in issues)
        assert any("deviceB.z: out of range" in i for i in issues)

    def test_skips_non_dict_top_level_values(self, tmp_path):
        ep = _build_episode(tmp_path, {"d": {"a": np.zeros(3)}})
        config = {"enabled": True, "d": {"a": {"no_nan": True}}}
        results, issues = run_numeric_value_checks(ep, config)
        # "enabled" is filtered, only device entries remain
        assert "enabled" not in results
        assert results == {"d": {"a": results["d"]["a"]}}
