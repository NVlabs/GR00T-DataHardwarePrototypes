# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive calibration script for DexUMI exoskeleton.

Calibrates zero_position_angle (and optionally directions), then saves them to
the device's config YAML together with the calibration provenance:
`calibration_method` (which reference pose produced the zeros) and, for tool
methods, `calibration_tool_angles` (the exact reference table used). The config also records `device_serial`, the USB serial
of the board it was calibrated on: recording finds the hand's port by it and
refuses any other board. A glove with no config yet starts from its side's design
template, config/dexumi_exoskeleton/templates/<device>.template.yaml.

Usage:
    # v4 calibration tool (with its hardstops):
    python examples/hand/exoskeleton/calibrate_exo.py --device DEXUMI-Right --version v4
    # Manual flat pose + direction sweep, for a glove built differently from its
    # design template (records the board, the only Pico board plugged in):
    python examples/hand/exoskeleton/calibrate_exo.py --device DEXUMI-Right --manual
    # Manual re-zero only, keeping the existing directions (implies --manual):
    python examples/hand/exoskeleton/calibrate_exo.py --device <name> --skip-directions
    # Bench override (skips name-based identity verification; the config then
    # records the board on that port):
    python examples/hand/exoskeleton/calibrate_exo.py --port /dev/ttyACM1 --config <yaml> --version v4
"""

from pathlib import Path

import click
import numpy as np
import yaml
from exo_device import (
    connected_board_port,
    device_options,
    open_encoder,
    resolve_config_path,
)

from umi.common.hardware_sdk.sensor import EncoderSensor
from umi.common.utility.rate_keeper import RateKeeper
from umi.utils.device_manager import get_exoskeleton_template_path

ENCODER_IDS = list(range(15))
NUM_JOINTS = 15
SAMPLE_COUNT = 30  # ~0.5s at 60Hz

# Calibration-tool joint targets are in degrees, indexed 0..14 (five fingers,
# three joints each). --version selects the reference pose used to solve zeros.
# Fingertip joints 5, 8, 11 and 14 use robot-hand angles; invert fingertip_poly_map
# to obtain exoskeleton angles before solving. Other targets are exoskeleton angles.

# Reference pose of the v4 calibration tool with its hardstops installed (v4 exos
# are always calibrated against the hardstops).
CALIBRATION_TOOL_V4_ANGLES = np.array(
    [
        5.94,  # Joint 0
        2.84,  # Joint 1
        3.26,  # Joint 2
        8.49,  # Joint 3
        10.99,  # Joint 4
        1.25,  # Joint 5   (fingertip: robot-space, inverted via poly map)
        -1.02,  # Joint 6
        9.02,  # Joint 7
        2.37,  # Joint 8   (fingertip: robot-space, inverted via poly map)
        -10.30,  # Joint 9
        7.22,  # Joint 10
        1.75,  # Joint 11  (fingertip: robot-space, inverted via poly map)
        -18.69,  # Joint 12
        7.34,  # Joint 13
        3.97,  # Joint 14  (fingertip: robot-space, inverted via poly map)
    ]
)

# Fingertip tool targets use robot-hand angles. The configured fingertip_poly_map
# provides the inversion coefficients; these indices support the missing-map warning.
FINGERTIP_JOINTS = (5, 8, 11, 14)

# Calibration-method identifiers written into the config as `calibration_method`:
# which reference pose the committed zeros were derived from.
METHOD_TOOL_V4 = "calibration_tool_v4"
METHOD_MANUAL = "manual"

# method id -> reference angles the tool holds the hand at
METHOD_ANGLES = {
    METHOD_TOOL_V4: CALIBRATION_TOOL_V4_ANGLES,
}

# method id -> human-readable name for prompts and reports
METHOD_NAMES = {
    METHOD_TOOL_V4: "v4 calibration tool (with hardstops)",
    METHOD_MANUAL: "manual flat pose",
}

# exo version (--version) <-> the calibration method of the tool built for it
VERSION_METHODS = {"v4": METHOD_TOOL_V4}
VERSION_CHOICES = tuple(VERSION_METHODS)
# To support another tool version, add its reference-angle table and method
# identifier to METHOD_ANGLES, METHOD_NAMES and VERSION_METHODS.


def load_exo_config(config_path: str | Path) -> dict:
    """Load the device config YAML as a plain dict ({} for an empty file)."""
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_directions(config: dict, config_path: str | Path) -> str:
    """The committed 'directions' string, required by the tool methods."""
    directions = config.get("directions")
    if not directions:
        raise SystemExit(
            f"The calibration tool needs a 'directions' field in {config_path}, "
            f"but none was found. Run the manual calibration first "
            f"(calibrate_exo.py --manual)."
        )
    return directions


def get_fingertip_poly_map(config: dict) -> dict[int, np.ndarray] | None:
    """Parse 'fingertip_poly_map' (exo_angle -> robot_angle polynomials).

    Returns {joint_index: coeffs} or None if absent.
    """
    poly_map = config.get("fingertip_poly_map")
    if not poly_map:
        return None
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in poly_map.items()}


def read_raw_angles(
    encoder: EncoderSensor, num_samples: int = SAMPLE_COUNT
) -> np.ndarray:
    """Read and average multiple encoder samples for stability."""
    samples = []
    rate_keeper = RateKeeper(target_rate=60, verbose=False)
    while len(samples) < num_samples:
        with rate_keeper:
            frame = encoder.get_frame()
            if frame is not None and frame.position is not None:
                samples.append(np.array(frame.position, dtype=np.float64))
    return np.mean(samples, axis=0)


def parse_direction_signs(directions: str) -> np.ndarray:
    """Convert a '+'/'-' direction string to an array of +1.0 / -1.0."""
    if len(directions) != NUM_JOINTS:
        raise ValueError(
            f"directions length ({len(directions)}) must match NUM_JOINTS "
            f"({NUM_JOINTS}): {directions!r}"
        )
    return np.array([1.0 if c == "+" else -1.0 for c in directions])


def invert_fingertip_poly(coeffs: np.ndarray, robot_angle: float) -> float:
    """Invert a fingertip polynomial: find exo angle x with polyval(coeffs, x) == robot_angle.

    `fingertip_poly_map` stores the forward map exo_angle -> robot_angle (degrees),
    applied at runtime in DexUmiExoskeleton.convert_to_robot_actions. A fingertip
    joint's calibration-tool target is given in robot-hand space, so we invert the
    polynomial to recover the exo-space signed angle the zero-position back-solve
    expects.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    shifted = coeffs.copy()
    shifted[-1] -= robot_angle  # solve polyval(coeffs, x) - robot_angle == 0
    roots = np.roots(shifted)
    real_roots = roots[np.abs(roots.imag) < 1e-6].real
    if real_roots.size == 0:
        raise ValueError(
            f"No real root inverting fingertip polynomial for robot angle "
            f"{robot_angle:.4f} (coeffs={coeffs.tolist()})"
        )
    # The map is near-linear and monotonic over each joint's range; pick the real
    # root nearest the linear estimate to reject spurious far-off cubic roots.
    linear_est = (robot_angle - coeffs[-1]) / coeffs[-2]
    return float(real_roots[np.argmin(np.abs(real_roots - linear_est))])


def calibrate_zero_manual(encoder: EncoderSensor) -> np.ndarray:
    """Record zero position angles when exoskeleton is flat."""
    print("\n" + "=" * 60)
    print("STEP 1: Zero Position Calibration")
    print("=" * 60)
    print("Flatten the exoskeleton to its zero/home position.")
    input("Press Enter when ready to record... ")

    print("Recording zero position (averaging samples)...")
    zero_angles = read_raw_angles(encoder)

    print("\nZero position angles recorded:")
    for i, angle in enumerate(zero_angles):
        print(f"  Joint {i:2d}: {angle:8.3f}°  ({np.deg2rad(angle):8.4f} rad)")

    return zero_angles


def calibrate_directions(encoder: EncoderSensor, zero_angles: np.ndarray) -> str:
    """Determine +/- direction for each joint."""
    print("\n" + "=" * 60)
    print("STEP 2: Direction Calibration")
    print("=" * 60)
    print("For each joint, rotate it in the POSITIVE direction, then press Enter.\n")

    directions = []
    for joint_idx in range(NUM_JOINTS):
        input(
            f"Joint {joint_idx:2d}: Rotate in positive direction, then press Enter... "
        )

        current_angles = read_raw_angles(encoder, num_samples=10)
        offset = (current_angles[joint_idx] - zero_angles[joint_idx] + 180) % 360 - 180

        sign = "+" if offset >= 0 else "-"
        directions.append(sign)
        print(
            f"  -> offset = {offset:+.3f}° ({np.deg2rad(offset):+.4f} rad) "
            f"-> direction = '{sign}'\n"
        )

    direction_str = "".join(directions)
    print(f"Directions: '{direction_str}'")
    return direction_str


def calibrate_zero_with_tool(
    encoder: EncoderSensor,
    directions: str,
    tool_angles: np.ndarray,
    method_name: str,
    fingertip_poly_map: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Derive zero positions from the calibration tool's known reference pose.

    `tool_angles` is the reference pose the tool holds (CALIBRATION_TOOL_V4_ANGLES
    for the v4 tool with its hardstops). Inverting the runtime formula
        joint_angle = ((raw - zero + 180) % 360 - 180) * direction_sign
    gives
        zero = (raw - joint_angle * direction_sign) % 360
    so the directions must already be known (loaded from the config).

    Fingertip joints (5, 8, 11, 14) are special: their `tool_angles`
    entry is a robot-hand angle, and the runtime maps exo_angle -> robot_angle via
    `fingertip_poly_map`. We invert that polynomial to recover the exo-space
    signed `joint_angle` before back-solving; all other joints are used as-is.
    """
    print("\n" + "=" * 60)
    print(f"Zero Position Calibration ({method_name})")
    print("=" * 60)
    print(f"Seat the exoskeleton fully into the {method_name}.")
    input("Press Enter when ready to record... ")

    print("Recording calibration-tool pose (averaging samples)...")
    raw_angles = read_raw_angles(encoder)

    signs = parse_direction_signs(directions)

    # Convert tool targets to exo-space signed angles. Fingertip joints are given
    # in robot-hand space, so invert their polynomial; all others are exo-space.
    exo_target_angles = np.asarray(tool_angles, dtype=np.float64).copy()
    inverted_joints = set()
    if fingertip_poly_map:
        for idx, coeffs in fingertip_poly_map.items():
            if 0 <= idx < NUM_JOINTS:
                exo_target_angles[idx] = invert_fingertip_poly(
                    coeffs, exo_target_angles[idx]
                )
                inverted_joints.add(idx)
    else:
        print(
            f"  WARNING: no fingertip_poly_map provided; fingertip joints "
            f"{FINGERTIP_JOINTS} treated as exo-space angles (no inversion)."
        )

    zero_angles = (raw_angles - exo_target_angles * signs) % 360

    print(f"\nZero position angles derived from {method_name}:")
    for i in range(NUM_JOINTS):
        if i in inverted_joints:
            target_note = (
                f"robot={tool_angles[i]:+7.3f}° "
                f"-> exo={exo_target_angles[i]:+7.3f}° (poly-inverted)"
            )
        else:
            target_note = f"target={exo_target_angles[i]:+7.3f}°"
        print(
            f"  Joint {i:2d}: "
            f"raw={raw_angles[i]:8.3f}°  "
            f"{target_note}  "
            f"dir='{directions[i]}'  "
            f"-> zero={zero_angles[i]:8.3f}° ({np.deg2rad(zero_angles[i]):8.4f} rad)"
        )

    return zero_angles


def save_config(
    config_path: str | Path,
    existing_config: dict | None,
    zero_angles: np.ndarray,
    directions: str | None,
    method: str,
    tool_angles: np.ndarray | None,
    device_serial: str | None = None,
) -> None:
    """Save calibration results to the config YAML, preserving existing fields.

    ``device_serial`` is the USB serial of the board just calibrated; the config
    records it so recording opens this hand on that board only.

    If ``directions`` is None, the existing ``directions`` field is left
    untouched (zero-position-only calibration). ``calibration_method`` records
    which reference pose produced these zeros (see ``METHOD_*``); for tool
    methods ``calibration_tool_angles`` stores the exact reference table used
    (full precision; FINGERTIP_JOINTS entries are robot-hand space), so the
    calibration stays reproducible even if the constants change later.
    """
    config_path = Path(config_path)

    if existing_config is not None:
        config = dict(existing_config)
        print(f"\nLoaded existing config from {config_path}")
    else:
        config = {
            "device_name": "dexumi_exoskeleton",
            "encoder_sensor_ids": ENCODER_IDS,
        }
        print(
            "\nNo existing config found; creating a minimal one. Fill in "
            "tactile_sensor_ids, fingertip_poly_map, exo_to_robot_joint_map, "
            "num_hand_joints, joint_limits, hand_config and geometry_verified "
            "before recording."
        )

    config["zero_position_angle"] = [round(float(v), 3) for v in zero_angles]
    if directions is not None:
        config["directions"] = directions
    elif "directions" in config:
        print(f"Preserving existing directions: '{config['directions']}'")
    if device_serial is not None:
        recorded = config.get("device_serial")
        if recorded and str(recorded).strip() != device_serial:
            print(
                f"NOTE: {config_path} recorded device_serial {recorded}; the board "
                f"changed, recording {device_serial}."
            )
        config["device_serial"] = device_serial
    config["calibration_method"] = method
    if tool_angles is not None:
        config["calibration_tool_angles"] = [float(v) for v in tool_angles]
    else:
        # Manual zeros have no tool reference; drop any stale table.
        config.pop("calibration_tool_angles", None)
    # Remove unsupported provenance keys so the saved calibration can be loaded.
    for key in ("calibration_pose", "calibration_status"):
        config.pop(key, None)

    print("\nConfig to save:")
    print(yaml.dump(config, default_flow_style=None))

    confirm = input(f"Save to {config_path}? [y/N] ").strip().lower()
    if confirm == "y":
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=None)
        print(f"Config saved to {config_path}")
    else:
        print("Config not saved.")


@click.command(help="Calibrate a DexUMI exoskeleton and save its config YAML.")
@device_options
@click.option(
    "--version",
    type=click.Choice(VERSION_CHOICES),
    default=None,
    help=(
        "Exo version: derive the zeros from its calibration tool's known pose "
        "using the existing directions (v4: the tool with its hardstops). "
        "Required unless --manual."
    ),
)
@click.option(
    "--manual",
    is_flag=True,
    help=(
        "Manual flat-pose calibration and direction sweep instead of the "
        "calibration tool, for a glove built differently from its design template."
    ),
)
@click.option(
    "--skip-directions",
    is_flag=True,
    help=(
        "Manual calibration of the zero position angle only; keep existing "
        "directions. Implies --manual."
    ),
)
def main(device, port, config_override, version, manual, skip_directions):
    manual = manual or skip_directions
    if manual and version:
        raise click.UsageError(
            "--manual / --skip-directions calibrate without the calibration tool; "
            "drop --version."
        )
    if not device and not port:
        raise SystemExit("Error: --device is required unless --port is given")
    if not manual and not version:
        raise click.UsageError(
            "name the exo version of the calibration tool (--version v4; "
            "the device name does not carry it), or pass --manual for the "
            "manual flat-pose calibration."
        )
    if manual:
        method = METHOD_MANUAL
    else:
        method = VERSION_METHODS[version]

    # Calibration may target a brand-new hand, so the canonical config need not
    # exist yet — save_config creates it.
    config_path = resolve_config_path(device, config_override, require_exists=False)
    print(f"Calibration config: {config_path}")
    existing_config = (
        load_exo_config(config_path) if Path(config_path).exists() else None
    )
    if existing_config is None and device and not config_override:
        # A glove's first calibration starts from its side's design template
        # (directions, fingertip map, joint limits), as in the GUI.
        template_path = get_exoskeleton_template_path(device)
        if Path(template_path).exists():
            existing_config = load_exo_config(template_path)
            print(f"{device} has no calibration yet; starting from {template_path}")

    # Load direction signs and fingertip mappings before opening hardware; both
    # are required to solve calibration zeros.
    tool_angles = None
    tool_directions = None
    tool_poly_map = None
    if method != METHOD_MANUAL:
        tool_angles = METHOD_ANGLES[method]
        if existing_config is None:
            raise SystemExit(
                f"The calibration tool needs existing directions, but "
                f"{config_path} does not exist. Run the manual calibration first "
                f"(calibrate_exo.py --manual)."
            )
        tool_directions = get_directions(existing_config, config_path)
        tool_poly_map = get_fingertip_poly_map(existing_config)
        if not tool_poly_map:
            print(
                f"WARNING: no 'fingertip_poly_map' in {config_path}; fingertip "
                f"joints {FINGERTIP_JOINTS} will be treated as exo-space angles."
            )

    # An uncalibrated glove has no saved board identity, so select the connected board.
    if device and not port and not (existing_config or {}).get("device_serial"):
        port = connected_board_port()
        print(f"{device}: no board recorded in {config_path} yet; using {port}.")

    print("Waiting for encoder data...")
    reader, encoder = open_encoder(
        sensor_ids=ENCODER_IDS,
        device=None if port else device,
        port=port,
        device_name="calibration_encoder",
    )
    board_serial = reader.get_serial_for_port(reader.port)

    try:
        if method != METHOD_MANUAL:
            zero_angles = calibrate_zero_with_tool(
                encoder,
                tool_directions,
                tool_angles,
                METHOD_NAMES[method],
                fingertip_poly_map=tool_poly_map,
            )
            directions = None  # preserve existing directions
        else:
            zero_angles = calibrate_zero_manual(encoder)
            if skip_directions:
                print("\nSkipping direction calibration (--skip-directions).")
                directions = None
            else:
                directions = calibrate_directions(encoder, zero_angles)
        save_config(
            config_path,
            existing_config,
            zero_angles,
            directions,
            method,
            tool_angles,
            board_serial,
        )
    except KeyboardInterrupt:
        print("\nCalibration cancelled.")
    finally:
        encoder.stop_streaming()
        reader.stop()


if __name__ == "__main__":
    main()
