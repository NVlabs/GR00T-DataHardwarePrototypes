# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import MagicMock

import pytest

from umi.umi_device.umi_gripper import (
    CLOSE_SPREAD_MAX_DEG,
    DEFAULT_GEAR_RADIUS_MM,
    OPEN_TRAVEL_MAX_DEG,
    OPEN_TRAVEL_MIN_DEG,
    GripperHardwareFault,
    UmiGripper,
)

# Choose test openings from the acceptance band, keeping the over-limit
# case below the 180-degree encoder wrap.
FULL_OPEN_DEG = (OPEN_TRAVEL_MIN_DEG + OPEN_TRAVEL_MAX_DEG) / 2.0
TOO_FAR_DEG = (OPEN_TRAVEL_MAX_DEG + 180.0) / 2.0


def make_gripper(zero=100.0, directions="+"):
    return UmiGripper(
        device_name="test_gripper",
        reader=MagicMock(),
        distance_zero_angle=zero,
        directions=directions,
    )


def mm(degrees):
    """The driver's model at the default gear radius: 2*pi*r*deg/360*2."""
    return 2 * math.pi * DEFAULT_GEAR_RADIUS_MM * degrees / 360.0 * 2


def raw_angle_to_distance(g, angle):
    """Mirror UmiGripper._get_encoder_frame: clip the signed offset at the
    closed position, then convert that joint angle to distance."""
    return g.angle_to_distance(max(0.0, g.get_signed_angle_offset(angle)))


@pytest.mark.parametrize(
    "directions,zero,current,expected",
    [
        ("+", 100, 130, 30.0),
        ("+", 100, 70, -30.0),
        ("+", 100, 100, 0.0),
        ("+", 350, 10, 20.0),
        ("+", 10, 350, -20.0),
        ("-", 100, 130, -30.0),
        ("-", 100, 70, 30.0),
        ("-", 350, 10, -20.0),
        ("-", 10, 350, 20.0),
    ],
)
def test_signed_offset(directions, zero, current, expected):
    g = make_gripper(zero=zero, directions=directions)
    assert g.get_signed_angle_offset(current) == pytest.approx(expected)


def test_180_boundary():
    g = make_gripper(zero=0.0, directions="+")
    assert abs(g.get_signed_angle_offset(180.0)) == pytest.approx(180.0)


@pytest.mark.parametrize("bad", ["++", "x", "", "+-", "++++"])
def test_invalid_directions(bad):
    with pytest.raises(ValueError):
        make_gripper(directions=bad)


def test_angle_to_distance_clips_negative():
    g = make_gripper(zero=100.0, directions="+")
    assert raw_angle_to_distance(g, 70.0) == 0.0
    assert raw_angle_to_distance(g, 130.0) == pytest.approx(mm(30.0))


def test_angle_to_distance_negative_direction():
    g = make_gripper(zero=100.0, directions="-")
    assert raw_angle_to_distance(g, 70.0) == pytest.approx(mm(30.0))
    assert raw_angle_to_distance(g, 130.0) == 0.0


def test_angle_to_distance_at_zero():
    g = make_gripper(zero=100.0, directions="+")
    assert raw_angle_to_distance(g, 100.0) == 0.0


def test_angle_to_distance_wrap():
    # zero near 0/360 wrap, current at 10 -> +20 deg opening for "+"
    g = make_gripper(zero=350.0, directions="+")
    assert raw_angle_to_distance(g, 10.0) == pytest.approx(mm(20.0))
    # opposite side wraps to negative -> clipped
    assert raw_angle_to_distance(g, 330.0) == 0.0


def test_directions_roundtrip_through_config():
    g_minus = make_gripper(directions="-")
    cfg = g_minus.get_config()
    assert cfg["directions"] == "-"
    assert cfg["distance_zero_angle"] == 100.0

    g_plus = make_gripper(directions="+")
    assert g_plus.get_config()["directions"] == "+"


def _mock_encoder_with_angles(angles):
    """Encoder mock whose get_frame() yields one angle per call (via .position[0])."""
    encoder = MagicMock()
    angle_iter = iter(angles)

    def next_frame():
        try:
            angle = next(angle_iter)
        except StopIteration:
            return None
        f = MagicMock()
        f.position = [angle]
        return f

    encoder.get_frame.side_effect = next_frame
    return encoder


# Every pair sits inside the acceptance band. The pairs exist to exercise the
# 0/360 wrap in both directions; their magnitude is FULL_OPEN_DEG, the middle of
# the band, so they keep clearing it as it moves.
@pytest.mark.parametrize(
    "close_angle,open_angle,expected_directions",
    [
        (100.0, 100.0 + FULL_OPEN_DEG, "+"),  # open positive of zero
        (200.0, 200.0 - FULL_OPEN_DEG, "-"),  # open negative of zero
        (350.0, (350.0 + FULL_OPEN_DEG) % 360.0, "+"),  # wrap forward
        (10.0, (10.0 - FULL_OPEN_DEG) % 360.0, "-"),  # wrap backward
    ],
)
def test_interactive_calibrate_simulates_close_then_open(
    monkeypatch, close_angle, open_angle, expected_directions
):
    # Construct with placeholders so __init__ skips calibration; the test drives it.
    g = make_gripper(zero=0.0, directions="+")

    NUM_SAMPLES_PER_CAPTURE = 10  # mirrors constant inside interactive_calibrate
    n_close = 1
    angles = [close_angle] * NUM_SAMPLES_PER_CAPTURE * n_close + [
        open_angle
    ] * NUM_SAMPLES_PER_CAPTURE
    g.encoder = _mock_encoder_with_angles(angles)

    # Three input() calls in order: close prompt, open prompt, save-config prompt.
    inputs = iter(["", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    # Skip the 0.01s sleeps inside the sampling loop.
    import umi.umi_device.umi_gripper as _ug

    monkeypatch.setattr(_ug.time, "sleep", lambda _s: None)

    ok = g.interactive_calibrate(n=n_close)

    assert ok is True
    assert g.distance_zero_angle == pytest.approx(close_angle)
    assert g.directions == expected_directions
    assert g.direction_sign == (1.0 if expected_directions == "+" else -1.0)
    # Sanity: opening produces positive distance after calibration regardless of side.
    assert raw_angle_to_distance(g, open_angle) > 0.0
    assert raw_angle_to_distance(g, close_angle) == pytest.approx(0.0)


# ------------------------------------------------- consistency checks + fault

CAPTURE = 10  # NUM_SAMPLES_PER_CAPTURE inside interactive_calibrate


def _drive(monkeypatch, gripper, captures, answers):
    """Feed one averaged angle per capture and one answer per prompt.

    `answers` may hold exception instances, which are raised instead of
    returned -- that is how the EOF-at-the-prompt case is expressed.
    """
    import umi.umi_device.umi_gripper as _ug

    stream = [angle for angle in captures for _ in range(CAPTURE)]
    gripper.encoder = _mock_encoder_with_angles(stream)

    supplied = iter(answers)

    def fake_input(_prompt=""):
        value = next(supplied)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(_ug.time, "sleep", lambda _s: None)
    # Otherwise a station's own auto-save path skips the save prompt and every
    # answer below lands one prompt early.
    monkeypatch.delenv("UMI_GRIPPER_AUTO_SAVE_PATH", raising=False)


def test_close_spread_within_limit_passes(monkeypatch):
    g = make_gripper()
    _drive(
        monkeypatch,
        g,
        [100.0, 100.5, 101.0, 100.5 + FULL_OPEN_DEG],
        ["", "", "", "", ""],
    )

    assert g.interactive_calibrate() is True
    assert g.distance_zero_angle == pytest.approx(100.5)
    assert g.directions == "+"


def test_close_spread_measured_across_the_wrap(monkeypatch):
    """359.5/0.2/359.8 are 0.7 apart, not 359.3 -- and average to 359.83."""
    g = make_gripper()
    _drive(monkeypatch, g, [359.5, 0.2, 359.8, FULL_OPEN_DEG], ["", "", "", "", ""])

    assert g.interactive_calibrate() is True
    assert g.distance_zero_angle == pytest.approx(359.8333, abs=1e-3)
    assert g.directions == "+"


def test_close_spread_beyond_limit_retries_from_the_start(monkeypatch):
    g = make_gripper()
    _drive(
        monkeypatch,
        g,
        # 7 deg apart, then a clean set, then the open capture.
        [100.0, 107.0, 101.0, 200.0, 200.5, 201.0, 200.5 + FULL_OPEN_DEG],
        ["", "", "", "", "", "", "", "", ""],
    )

    assert g.interactive_calibrate() is True
    # Only the second set feeds the zero; a retry discards what came before.
    assert g.distance_zero_angle == pytest.approx(200.5)


def test_open_travel_below_limit_retries(monkeypatch):
    g = make_gripper()
    _drive(
        monkeypatch,
        g,
        # 20 deg of travel is under the minimum; the retry opens fully.
        [100.0, 100.0, 100.0, 120.0, 100.0, 100.0, 100.0, 100.0 + FULL_OPEN_DEG],
        ["", "", "", "", "", "", "", "", "", ""],
    )

    assert g.interactive_calibrate() is True
    assert g.distance_zero_angle == pytest.approx(100.0)
    assert g.directions == "+"


def test_open_travel_above_limit_retries(monkeypatch):
    """Travel past the band means the closes were not on the hard stop.

    Three closes that agree with each other but sit short of the mechanical
    stop pass the repeatability check and still put the zero in the wrong
    place; the only evidence left is that the open then reads implausibly far
    from it."""
    g = make_gripper()
    _drive(
        monkeypatch,
        g,
        # TOO_FAR_DEG of travel is over the maximum; the retry opens fully.
        [
            100.0,
            100.0,
            100.0,
            100.0 + TOO_FAR_DEG,
            100.0,
            100.0,
            100.0,
            100.0 + FULL_OPEN_DEG,
        ],
        ["", "", "", "", "", "", "", "", "", ""],
    )

    assert g.interactive_calibrate() is True
    assert g.distance_zero_angle == pytest.approx(100.0)
    assert g.directions == "+"


def test_eof_at_the_prompt_retries_rather_than_propagating(monkeypatch):
    g = make_gripper()
    _drive(
        monkeypatch,
        g,
        [100.0, 107.0, 101.0, 200.0, 200.5, 201.0, 200.5 + FULL_OPEN_DEG],
        ["", "", "", EOFError(), "", "", "", "", ""],
    )

    assert g.interactive_calibrate() is True
    assert g.distance_zero_angle == pytest.approx(200.5)


def test_reporting_a_fault_raises_and_saves_nothing(monkeypatch):
    g = make_gripper(zero=100.0)
    g.save_config = MagicMock()
    _drive(monkeypatch, g, [200.0, 207.0, 201.0], ["", "", "", "f"])

    with pytest.raises(GripperHardwareFault) as raised:
        g.interactive_calibrate()

    # The exception carries the same sentence the operator just read, mm and
    # all -- it renders from the payload rather than from the live gripper, so
    # it is the one place the two could drift apart.
    assert "7.00\u00b0 (~3.7 mm)" in str(raised.value)

    payload = raised.value.payload
    assert payload["check"] == "close_repeatability"
    assert payload["measured_deg"] == pytest.approx(7.0)
    assert payload["limit_deg"] == pytest.approx(CLOSE_SPREAD_MAX_DEG)
    assert payload["open_angle_deg"] is None
    assert payload["attempts"] == 1

    g.save_config.assert_not_called()
    # The zero it was constructed with survives: a fault leaves the gripper
    # exactly as uncalibrated as it found it.
    assert g.distance_zero_angle == pytest.approx(100.0)
    g.encoder.stop_streaming.assert_called()


def test_reporting_a_travel_fault_carries_the_open_angle(monkeypatch):
    g = make_gripper()
    _drive(monkeypatch, g, [100.0, 100.0, 100.0, 120.0], ["", "", "", "", "f"])

    with pytest.raises(GripperHardwareFault) as raised:
        g.interactive_calibrate()

    payload = raised.value.payload
    assert payload["check"] == "open_travel"
    assert payload["measured_deg"] == pytest.approx(20.0)
    assert payload["limit_deg"] == pytest.approx(OPEN_TRAVEL_MIN_DEG)
    assert payload["open_angle_deg"] == pytest.approx(120.0)


def test_reporting_a_high_travel_fault_names_the_upper_limit(monkeypatch):
    """The two sides of the band are separate checks in the report.

    A fault filed under one id and a limit that belongs to the other would
    read as the opposite failure."""
    g = make_gripper()
    _drive(
        monkeypatch,
        g,
        [100.0, 100.0, 100.0, 100.0 + TOO_FAR_DEG],
        ["", "", "", "", "f"],
    )

    with pytest.raises(GripperHardwareFault) as raised:
        g.interactive_calibrate()

    payload = raised.value.payload
    assert payload["check"] == "open_travel_high"
    assert payload["measured_deg"] == pytest.approx(TOO_FAR_DEG)
    assert payload["limit_deg"] == pytest.approx(OPEN_TRAVEL_MAX_DEG)
    assert payload["open_angle_deg"] == pytest.approx(100.0 + TOO_FAR_DEG)


@pytest.mark.parametrize(
    "gear_radius,spread_mm,limit_mm",
    [
        (DEFAULT_GEAR_RADIUS_MM, "3.7", "1.0"),  # the default: 0.524 mm per degree
        (29.0, "7.1", "2.0"),  # a gripper configured otherwise
    ],
)
def test_failure_message_reports_mm_beside_degrees(
    monkeypatch, capsys, gear_radius, spread_mm, limit_mm
):
    """Degrees are what the encoder reports; mm is what the operator can judge
    against the jaw in front of them. The conversion has to come from the
    gripper's own configured gear radius -- a hardcoded default would quietly
    misreport on any gripper calibrated with a different one."""
    g = make_gripper()
    g.gear_radius = gear_radius
    _drive(
        monkeypatch,
        g,
        [100.0, 107.0, 101.0, 200.0, 200.5, 201.0, 200.5 + FULL_OPEN_DEG],
        ["", "", "", "", "", "", "", "", ""],
    )

    assert g.interactive_calibrate() is True

    out = capsys.readouterr().out
    assert f"7.00\u00b0 (~{spread_mm} mm)" in out
    assert f"2.00\u00b0 (~{limit_mm} mm)" in out


# --------------------------------------------------------------------------
# the V4 flexure gripper's geometry: a 15 mm pinion
# --------------------------------------------------------------------------


def make_v4_gripper(zero=25.0):
    return UmiGripper(
        device_name="v4_gripper",
        reader=MagicMock(),
        gripper_generation="v4",
        distance_zero_angle=zero,
        directions="+",
    )


def test_a_v4_gripper_takes_its_radius_and_band_from_the_generation():
    g = make_v4_gripper()

    assert g.gear_radius == 15.0
    assert (g.open_travel_min_deg, g.open_travel_max_deg) == (120.0, 175.0)
    # A V4 prototype measured ~148 deg of travel for a ~78 mm opening.
    assert g.angle_to_distance(148.0) == pytest.approx(77.5, abs=0.1)
    saved = g.get_config()
    assert saved["gripper_generation"] == "v4"
    assert saved["gear_radius"] == 15.0


def test_without_a_generation_the_v4_defaults_apply_and_nothing_new_is_saved():
    """A calibration file can predate generations. A gripper built from one
    takes the V4 geometry, and saving it must not grow a key the file never
    had."""
    g = make_gripper()

    assert g.gear_radius == 15.0
    assert (g.open_travel_min_deg, g.open_travel_max_deg) == (
        OPEN_TRAVEL_MIN_DEG,
        OPEN_TRAVEL_MAX_DEG,
    )
    assert "gripper_generation" not in g.get_config()


def test_an_explicit_radius_beats_the_generation():
    g = UmiGripper(
        device_name="x",
        reader=MagicMock(),
        gripper_generation="v4",
        gear_radius=15.5,
        distance_zero_angle=0.0,
        directions="+",
    )
    assert g.gear_radius == 15.5


def test_an_unknown_generation_is_refused():
    with pytest.raises(ValueError, match="unknown gripper_generation"):
        UmiGripper(
            device_name="x",
            reader=MagicMock(),
            gripper_generation="v9",
            distance_zero_angle=0.0,
            directions="+",
        )


def test_v4_calibration_accepts_the_prototypes_open_on_the_first_try(monkeypatch):
    """148 deg is what a V4 prototype measured fully open; it must pass on the
    first try."""
    g = make_v4_gripper()
    _drive(monkeypatch, g, [25.0, 25.0, 25.0, 173.0], ["", "", "", "", ""])

    assert g.interactive_calibrate() is True
    assert g.distance_zero_angle == pytest.approx(25.0)
    assert g.directions == "+"


def test_v4_calibration_still_rejects_a_two_thirds_open(monkeypatch):
    g = make_v4_gripper()
    _drive(
        monkeypatch,
        g,
        # 100 deg is two-thirds of the V4 stroke: under the floor, so it
        # retries; the retry opens fully.
        [25.0, 25.0, 25.0, 125.0, 25.0, 25.0, 25.0, 173.0],
        ["", "", "", "", "", "", "", "", "", ""],
    )

    assert g.interactive_calibrate() is True
    assert g.distance_zero_angle == pytest.approx(25.0)
