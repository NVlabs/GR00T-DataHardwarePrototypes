# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example: Using UmiGripper with angle-to-distance conversion.

This example demonstrates:
1. Setting up a UMI gripper with encoder sensor
2. Interactive calibration for distance measurement
3. Reading gripper angles and distances
4. Saving and loading config data
5. Loading gripper from config file
6. Recording data and performing online validation checks
7. Visualizing gripper data with jump detection

Usage:
    python umi_gripper_example.py --port /dev/ttyACM2
    python umi_gripper_example.py --uart-device-name DEADBEEFDEADBEEF
    python umi_gripper_example.py --port /dev/ttyACM2 --duration 10
    python umi_gripper_example.py --port /dev/ttyACM2 --no-record  # Streaming mode only
    python umi_gripper_example.py --from-config --config config/umi_gripper/gripper_config.yaml
"""

import json
import os
import sys
import time

import click
import numpy as np

from umi.common.hardware_sdk.sensor import UartSensorReader
from umi.common.utility import gripper_fault_report
from umi.common.utility.gripper_pipeline import GripperPipeline
from umi.common.utility.rate_keeper import RateKeeper
from umi.umi_device.umi_gripper import GripperHardwareFault, UmiGripper
from umi.utils.device_manager import normalize_umi_gripper_identifier

# Exit codes: 0 calibrated, 1 an ordinary error, 3 the operator reported a
# hardware fault. The Station Console reads this status from the file its
# terminal wrapper writes (`station_ui/calibration.py`).
HARDWARE_FAULT_EXIT = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from real_scripts.data_processing.data_check import (
    check_gripper_jumps,
    check_gripper_movement,
    check_gripper_range,
    visualize_gripper_data,
)

DEFAULT_CONFIG_FILE = os.path.join(
    PROJECT_ROOT, "config", "umi_gripper", "gripper_config.yaml"
)


def _uart_reader_kwargs(port: str, uart_device_name: str | None) -> dict:
    """Use identity-based resolution when calibration receives an identifier."""
    if uart_device_name:
        return {
            "uart_device_name": normalize_umi_gripper_identifier(uart_device_name),
            "baudrate": 1_000_000,
        }
    return {"port": port, "baudrate": 1_000_000}


# =============================================================================
# Display Functions
# =============================================================================


# Use the plot filename convention from visualize_gripper_data so callers
# can locate output without parsing terminal logs.
PLOT_EPISODE_NAME = "Live Recording"
PLOT_DEVICE_NAME = "umi_gripper"


def plot_path(plot_dir: str) -> str:
    return os.path.join(plot_dir, f"{PLOT_EPISODE_NAME}_{PLOT_DEVICE_NAME}.png")


def _dump_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def report_hardware_fault(
    fault: GripperHardwareFault,
    gripper_identifier: str | None,
    summary_json: str | None,
) -> None:
    """Persist the fault and tell both readers what happened.

    Two audiences: the operator standing at the terminal, and the station
    console, which cannot read this terminal back and only ever sees
    `--summary-json`. The recording pass never runs on this path, so that file
    carries the fault instead of the usual movement/jump/range numbers.
    """
    report = gripper_fault_report.build(
        fault.payload, gripper_identifier=gripper_identifier
    )
    local_path = gripper_fault_report.write_local(report)

    click.echo("\n" + "=" * 60)
    click.echo("HARDWARE FAULT REPORTED")
    click.echo("=" * 60)
    click.echo(f"   {fault}")
    click.echo(f"   Report written to {local_path}")
    click.echo("   The gripper was NOT calibrated and no config was saved.")
    click.echo("=" * 60)

    if summary_json:
        _dump_json(
            summary_json,
            {
                "hardware_fault": {
                    **report,
                    "local_path": str(local_path),
                }
            },
        )


def write_summary_json(
    path: str, movement_result: dict, jumps_result: dict, range_result: dict
) -> None:
    """The three checks, for a caller reading a file rather than a terminal.

    The station console runs this in a terminal emulator it cannot read back,
    so the verdict has to land somewhere. Only the summary numbers: the frame
    lists behind them can run to thousands of entries and the plot already
    shows where they are.
    """
    summary = {
        "moved": bool(movement_result.get("moved")),
        "range_mm": movement_result.get("range_mm"),
        "min_mm": movement_result.get("min_mm"),
        "max_mm": movement_result.get("max_mm"),
        "movement_error": movement_result.get("error"),
        "num_jumps": jumps_result.get("num_jumps"),
        "has_jumps": bool(jumps_result.get("has_jumps")),
        "in_range": bool(range_result.get("in_range")),
        "expected_range": range_result.get("expected_range"),
        "num_out_of_range": range_result.get("num_out_of_range"),
    }
    _dump_json(path, summary)


def print_validation_results(
    movement_result: dict, jumps_result: dict, range_result: dict
) -> None:
    """Print validation results in a formatted way."""
    click.echo("\n" + "=" * 60)
    click.echo("DATA VALIDATION RESULTS")
    click.echo("=" * 60)

    click.echo("\n1. Movement Check:")
    if movement_result.get("error"):
        click.echo(f"   ❌ {movement_result['error']}")
    else:
        status = "✓" if movement_result["moved"] else "✗"
        click.echo(f"   {status} Moved: {movement_result['moved']}")
        click.echo(f"   Range: {movement_result['range_mm']:.2f} mm")
        click.echo(f"   Min: {movement_result['min_mm']:.2f} mm")
        click.echo(f"   Max: {movement_result['max_mm']:.2f} mm")

    click.echo("\n2. Jump Detection:")
    status = "✗" if jumps_result["has_jumps"] else "✓"
    click.echo(f"   {status} Jumps detected: {jumps_result['num_jumps']}")
    if jumps_result["has_jumps"]:
        click.echo(f"   Threshold: {jumps_result['jump_threshold_mm_per_s']:.0f} mm/s")
        for jump in jumps_result["jump_frames"][:5]:  # Show first 5 jumps
            click.echo(
                f"     - Frame {jump['frame_index']}: "
                f"{jump['before_value']} → {jump['after_value']} mm "
                f"({jump['magnitude_mm_per_s']:.0f} mm/s)"
            )
        if len(jumps_result["jump_frames"]) > 5:
            click.echo(f"     ... and {len(jumps_result['jump_frames']) - 5} more")

    click.echo("\n3. Range Validation:")
    status = "✓" if range_result["in_range"] else "✗"
    click.echo(f"   {status} All values in range: {range_result['in_range']}")
    click.echo(f"   Expected range: {range_result['expected_range']} mm")
    if not range_result["in_range"]:
        click.echo(f"   Out of range frames: {range_result['num_out_of_range']}")
        for oor in range_result["out_of_range_frames"][:5]:  # Show first 5
            click.echo(f"     - Frame {oor['frame_index']}: {oor['value']} mm")
        if len(range_result["out_of_range_frames"]) > 5:
            click.echo(
                f"     ... and {len(range_result['out_of_range_frames']) - 5} more"
            )

    click.echo("=" * 60)


def _warn_if_saved_config_disagrees(gripper, gripper_generation, gear_radius) -> None:
    """A saved calibration carries its own radius; say so when the caller
    asked for another, because --from-config keeps the file's value."""
    wanted = gear_radius
    if wanted is None and gripper_generation:
        wanted = GripperPipeline.parse(gripper_generation).gear_radius_mm
    if wanted is not None and abs(float(wanted) - float(gripper.gear_radius)) > 1e-6:
        click.secho(
            f"WARNING: the saved config's gear radius is {gripper.gear_radius} mm, "
            f"not the {wanted} mm requested. Its distances are computed for a "
            "different gripper generation; run without --from-config to "
            "recalibrate and replace it.",
            fg="yellow",
        )


@click.command()
@click.option("-p", "--port", default="/dev/ttyACM1", help="Serial port")
@click.option(
    "--uart-device-name",
    default=None,
    help="Registered gripper name or raw 16-hex USB serial (identity checked)",
)
@click.option(
    "--gripper-generation",
    type=click.Choice(GripperPipeline.values()),
    default=None,
    help=(
        "Which gripper this is, written into the saved config. Sets the gear "
        "radius the distances are computed with and the open-travel band the "
        "calibration accepts: v4 is a 15 mm pinion opening 120-175 deg. Without "
        "it the same V4 values apply."
    ),
)
@click.option(
    "--gear-radius",
    type=float,
    default=None,
    help="Pinion pitch radius in mm; overrides the generation's value.",
)
@click.option("-c", "--config", default=DEFAULT_CONFIG_FILE, help="Config file path")
@click.option(
    "--from-config",
    is_flag=True,
    help="Create gripper from config file using from_config()",
)
@click.option(
    "--record/--no-record",
    default=True,
    help="Record data and perform validation checks (default: True)",
)
@click.option(
    "--duration",
    type=float,
    default=None,
    help="Recording duration in seconds (default: record until Ctrl+C)",
)
@click.option(
    "--visualize/--no-visualize",
    default=True,
    help="Show visualization plot after recording (default: True, only applies when recording)",
)
@click.option(
    "--plot-dir",
    default=None,
    help=(
        "Save the plot here instead of opening a window. An interactive plot "
        "blocks the process until somebody closes it, which strands a caller "
        "that is waiting on the exit code (the station console does)."
    ),
)
@click.option(
    "--summary-json",
    default=None,
    help="Write the validation results here as JSON, for a non-interactive caller.",
)
def main(
    port: str,
    uart_device_name: str | None,
    gripper_generation: str | None,
    gear_radius: float | None,
    config: str,
    from_config: bool,
    record: bool,
    duration: float,
    visualize: bool,
    plot_dir: str | None,
    summary_json: str | None,
):
    """Main example with full gripper workflow."""
    reader = UartSensorReader(**_uart_reader_kwargs(port, uart_device_name))
    reader.start()

    try:
        # Constructing an uncalibrated gripper starts interactive calibration.
        # Stop immediately if the operator reports a hardware fault: there is no
        # valid calibration to use for measurement or recording.
        try:
            if from_config and os.path.exists(config):
                click.echo(f"Creating gripper from config: {config}")
                gripper = UmiGripper.from_config(config, reader=reader)
                _warn_if_saved_config_disagrees(
                    gripper, gripper_generation, gear_radius
                )
            else:
                gripper = UmiGripper(
                    device_name="umi_gripper",
                    reader=reader,
                    gear_radius=gear_radius,
                    gripper_generation=gripper_generation,
                )
        except GripperHardwareFault as fault:
            report_hardware_fault(fault, uart_device_name, summary_json)
            sys.exit(HARDWARE_FAULT_EXIT)

        gripper_source = gripper.get_data_sources()[0]
        gripper_source.start_streaming()
        time.sleep(0.5)  # Wait for initial data

        click.echo("=" * 60)
        mode_str = "Recording Mode" if record else "Streaming Mode"
        click.echo(f"UMI Gripper Example - {mode_str}")
        click.echo("=" * 60)

        click.echo("\n1. Reading gripper frame...")
        frame = gripper_source.get_frame()
        if frame and frame.joint_angles is not None:
            click.echo(f"   Angle: {frame.joint_angles[0]:.2f}°")
            click.echo(f"   Distance: {frame.distance:.2f} mm")
        else:
            click.echo("   No frame data available")

        click.echo("\n3. Current config:")
        cfg = gripper.get_config()
        for key, value in cfg.items():
            click.echo(f"   {key}: {value}")

        if record:
            click.echo(
                "\n4. Recording gripper data... Please aggressively use grippers"
            )
            if duration:
                click.echo(f"   Recording for {duration} seconds...")
            else:
                click.echo("   Press Ctrl+C to stop recording\n")

            recorded_joint_angles = []
            recorded_distances = []
            recorded_timestamps = []
            start_time = time.time()

            rate_keeper = RateKeeper(target_rate=30)  # 30 Hz
            try:
                while True:
                    with rate_keeper:
                        frame = gripper_source.get_frame()
                        if (
                            frame
                            and frame.distance is not None
                            and frame.joint_angles is not None
                        ):
                            current_time = time.time() - start_time
                            recorded_joint_angles.append(frame.joint_angles[0])
                            recorded_distances.append(frame.distance)
                            # Jump detection scores mm/s, so it needs the real
                            # sample spacing. Prefer the frame's own clock -- the
                            # same one the recorder stores as receive_time.
                            recorded_timestamps.append(
                                frame.receive_time
                                if frame.receive_time is not None
                                else time.time()
                            )

                            click.echo(
                                f"\r   [{current_time:6.2f}s] Distance: {frame.distance:6.2f} mm | Angle: {frame.joint_angles[0]:6.2f}° | Frames: {len(recorded_joint_angles)}",
                                nl=False,
                            )

                        if duration and (time.time() - start_time) >= duration:
                            break
            except KeyboardInterrupt:
                click.echo("\n\n   Stopped by user")

            click.echo(f"\n\n   Total frames recorded: {len(recorded_joint_angles)}")

            if len(recorded_joint_angles) > 0:
                joint_angles_array = np.array(recorded_joint_angles)
                distances_array = np.array(recorded_distances)
                timestamps_array = np.array(recorded_timestamps)

                movement_result = check_gripper_movement(distances_array)
                jumps_result = check_gripper_jumps(distances_array, timestamps_array)
                range_result = check_gripper_range(distances_array)

                print_validation_results(movement_result, jumps_result, range_result)

                # Machine-readable, for a caller that is not reading the terminal
                if summary_json:
                    write_summary_json(
                        summary_json, movement_result, jumps_result, range_result
                    )

                if visualize:
                    click.echo("\nGenerating visualization...")
                    # With --plot-dir the plot is saved and the process exits;
                    # without it, plt.show() blocks until a window is closed.
                    visualize_gripper_data(
                        joint_positions=joint_angles_array,
                        distance=distances_array,
                        episode_name=PLOT_EPISODE_NAME,
                        device_name=PLOT_DEVICE_NAME,
                        jump_frames=jumps_result["jump_frames"],
                        output_dir=plot_dir,
                    )
                    if plot_dir:
                        click.echo(f"   Plot written to {plot_path(plot_dir)}")
            else:
                click.echo("   No data recorded!")

        else:
            click.echo("\n4. Streaming gripper distance...")
            click.echo("   Press Ctrl+C to stop\n")

            rate_keeper = RateKeeper(target_rate=20)  # 20 Hz
            try:
                while True:
                    with rate_keeper:
                        frame = gripper_source.get_frame()
                        if frame and frame.distance is not None:
                            click.echo(
                                f"\r   Distance: {frame.distance:6.2f} mm | Angle: {frame.joint_angles[0]:6.2f}°",
                                nl=False,
                            )
                        else:
                            click.echo("\r   No frame data", nl=False)
            except KeyboardInterrupt:
                click.echo("\n\n   Stopped by user")

    finally:
        click.echo("\n" + "=" * 60)
        click.echo("Shutting down...")
        if "gripper_source" in locals():
            gripper_source.stop_streaming()
        reader.stop()
        click.echo("✓ Done")


if __name__ == "__main__":
    main()
