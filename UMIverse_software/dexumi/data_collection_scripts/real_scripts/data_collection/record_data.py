# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Composable Data Recording Script

Device binding: once every device is open the recorder writes
``<session>/device_binding.yaml`` (which physical device stands behind each
recorded source; see umi/common/recording/device_identity.py). The DC GUI
passes its Identify scan with ``--device-binding <yaml>``; a scanned device
that differs from the one actually opened refuses the launch (exit 3) before
the ready banner. Without the flag nothing is refused.

Usage:
    uv run real_scripts/data_collection/record_data.py --config config.yaml

Controls:
    s - Start recording
    w - Stop and save recording
    a - Restart episode (discard current)
    q - Quit
"""

import os
import time
from typing import Callable, Optional

import click
import cv2
import numpy as np
import yaml

from umi.common.data.frame import NumericFrame
from umi.common.recording.base_recorder import PENDING_SUFFIX
from umi.common.recording.device_identity import (
    build_device_binding,
    collect_identities,
    config_source_names,
    load_scan_binding,
    write_device_binding,
)
from umi.common.recording.record_manager import RecorderManager
from umi.common.recording.session_slot import (
    resolve_session_slot,
    save_launch_config,
)
from umi.common.utility.background_aligner import attach_background_services
from umi.common.utility.data_visualizer import DataVisualizer, VisualizationConfig
from umi.common.utility.log_capture import tee_stdio_to_file
from umi.common.utility.print_utils import Colors
from umi.common.utility.rate_keeper import RateKeeper
from umi.common.utility.recording_utils import (
    assert_all_uart_readers_connected,
    build_frame_definitions,
    create_numeric_recorders,
    create_video_recorders,
    load_config,
    stop_all_uart_readers,
)


# =============================================================================
# Main Recording Logic
# =============================================================================


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="config/data_recording/dexumi/bimanual_dexumi_config.yaml",
    help="Path to configuration YAML file",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(),
    default=None,
    help="Override data directory from config",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--device-binding",
    "device_binding",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Device scan handed over by the DC GUI (schema-1 yaml). A scanned "
    "device that differs from the device actually opened refuses the launch. "
    "Without it the launch binds the configured devices (bound_by: config).",
)
def main(
    config: str,
    data_dir: Optional[str],
    verbose: bool,
    device_binding: Optional[str] = None,
):
    """Composable data recording with configurable video and numeric sources."""
    device_binding_scan = None
    if device_binding:
        try:
            device_binding_scan = load_scan_binding(device_binding)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise click.UsageError(f"Invalid --device-binding: {exc}") from exc

    cfg = load_config(config)
    recording_cfg = cfg.get("recording", {})
    viz_cfg = cfg.get("visualization", {})

    if data_dir:
        recording_cfg["data_dir"] = data_dir
    if verbose:
        recording_cfg["verbose"] = True

    # Resolve and persist a new session suffix if acquisition settings or
    # calibrations conflict with the existing folder. Save the original path
    # before resolution mutates cfg so the rollover can be logged afterward.
    orig_data_path = os.path.expanduser(
        str(recording_cfg.get("data_dir", "./data")).rstrip("/")
    )
    data_path = resolve_session_slot(cfg, os.path.expanduser(config))
    os.makedirs(data_path, exist_ok=True)

    # Everything below is teed into <data_path>/record_data.log (append mode,
    # visible session dividers).
    log_path = os.path.join(data_path, "record_data.log")
    with tee_stdio_to_file(log_path):
        # Save the unexpanded launch config before adding per-source defaults.
        # Session conflict detection compares this snapshot with the next launch.
        save_launch_config(data_path, cfg)
        if data_path != orig_data_path:
            click.echo(
                f"{Colors.YELLOW}Config changed vs the previous session; "
                f"recording into a new slot: {data_path} "
                f"(task_name={cfg.get('meta', {}).get('task_name')}){Colors.RESET}"
            )
        _run_session(
            cfg,
            recording_cfg,
            viz_cfg,
            data_path,
            device_binding_scan=device_binding_scan,
        )


# Exit status of a launch refused because a device changed since the DC GUI's
# Identify scan (distinct from crashes and usage errors).
DEVICE_BINDING_REFUSED_EXIT_CODE = 3


def _bind_launch_devices(
    recorder_manager, data_path: str, scan: Optional[dict]
) -> Optional[str]:
    """Bind every recorded source to the device actually opened.

    Writes ``<session>/device_binding.yaml`` and hands the binding to the
    manager, which copies it into every episode. Returns the refusal message
    when a device differs from the DC GUI's scan, else None. Without a scan
    this never refuses; a write failure is a warning.
    """
    recorders = getattr(recorder_manager, "recorders", None) or []
    try:
        binding, mismatches, notes = build_device_binding(
            collect_identities(recorders), config_source_names(recorders), scan
        )
    except Exception as exc:  # noqa: BLE001 - provenance never blocks a launch
        click.echo(f"{Colors.YELLOW}[device binding] skipped: {exc}{Colors.RESET}")
        return None
    for note in notes:
        click.echo(f"{Colors.YELLOW}[device binding] {note}{Colors.RESET}")
    if mismatches:
        lines = "\n".join(f"  {line}" for line in mismatches)
        return (
            "[device binding] Launch refused -- the device behind a slot changed "
            f"since the DC GUI's Identify scan:\n{lines}\n"
            "Run Identify cameras again (or fix the cabling), then start again."
        )
    recorder_manager.device_binding = binding
    try:
        path = write_device_binding(data_path, binding)
    except Exception as exc:  # noqa: BLE001 - provenance never blocks a launch
        click.echo(
            f"{Colors.YELLOW}[device binding] could not write "
            f"{data_path}/device_binding.yaml: {exc}{Colors.RESET}"
        )
        return None
    click.echo(
        f"[device binding] {len(binding['sources'])} source(s) bound "
        f"({binding['bound_by']}) -> {path}"
    )
    return None


def _finalize_episode(
    recorder_manager: RecorderManager, episode_id: int
) -> Optional[list[str]]:
    """Finalize the saved staging episode. Returns issues on failure, else None.

    Never discards anything: a bookkeeping failure (metadata write,
    registration) must leave the saved recording on disk,
    either still in ``episode_N.pending`` or already renamed to ``episode_N``.
    Both paths are captured first because a post-rename failure has already
    advanced the recorder's episode ID.
    """
    pending_path = recorder_manager.get_pending_episode_path()
    final_path = recorder_manager.get_final_episode_path()
    try:
        if recorder_manager.finalize_episode():
            return None
        issues = ["Episode data could not be written to disk"]
    except Exception as exc:
        click.echo(
            f"{Colors.RED}Failed to finalize episode {episode_id}: {exc}{Colors.RESET}"
        )
        issues = [str(exc)]
    if os.path.isdir(final_path):
        issues.append(
            f"episode_{episode_id} is finalized locally; restart data collection "
            "to align it"
        )
    elif os.path.isdir(pending_path):
        issues.append(f"episode_{episode_id}{PENDING_SUFFIX} kept on disk for recovery")
    return issues


def _run_cleanup_steps(steps: list[tuple[str, Callable[[], object]]]) -> None:
    """Run every shutdown step; log failures and re-raise the first one at the end."""
    first_error: Optional[Exception] = None
    for label, step in steps:
        try:
            step()
        except Exception as exc:
            click.echo(f"{Colors.RED}[cleanup] {label} failed: {exc}{Colors.RESET}")
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _save_error_state(issues: list[str], *, keep_staging=False) -> dict:
    state = {
        "title": "SAVE ERROR: Failed to finalize episode",
        "issues": list(issues),
        "fix_hint": "Check disk space and permissions -- press R to resume",
    }
    if keep_staging:
        state["fix_hint"] = (
            "Press Q to quit; recover the retained take before restarting."
        )
        state["fatal"] = True
    return state


def _run_session(
    cfg: dict,
    recording_cfg: dict,
    viz_cfg: dict,
    data_path: str,
    device_binding_scan: Optional[dict] = None,
):
    """Main recording loop body. Separated so ``main`` can wrap it cleanly in
    the stdio-tee context manager."""
    click.echo(f"Session log: {os.path.join(data_path, 'record_data.log')}")

    default_record_fps = recording_cfg.get("record_fps", 30)
    default_stream_fps = recording_cfg.get("stream_fps", 60)
    default_video_frame_type = recording_cfg.get("video_frame_type", "RGBCameraFrame")
    default_numeric_frame_type = recording_cfg.get("numeric_frame_type", "NumericFrame")
    is_verbose = recording_cfg.get("verbose", False)
    video_encoder = recording_cfg.get("video_encoder", "ffmpeg_h264")
    require_all_sources = bool(recording_cfg.get("require_all_sources", False))

    try:
        video_recorders, _, expanded_video_configs = create_video_recorders(
            video_source_configs=cfg.get("video_sources", []),
            record_path=data_path,
            default_record_fps=default_record_fps,
            default_stream_fps=default_stream_fps,
            default_frame_type=default_video_frame_type,
            verbose=is_verbose,
            info_verbose=True,  # Always show video recorder info
            warning_verbose=True,  # Always show video recorder warnings
            video_encoder=video_encoder,
            require_all_sources=require_all_sources,
        )

        numeric_recorders, _, expanded_numeric_configs = create_numeric_recorders(
            numeric_source_configs=cfg.get("numeric_sources", []),
            record_path=data_path,
            default_record_fps=default_record_fps,
            default_stream_fps=default_stream_fps,
            default_frame_type=default_numeric_frame_type,
            verbose=is_verbose,
            info_verbose=True,
            warning_verbose=True,
            require_all_sources=require_all_sources,
        )
    except Exception:
        # Source construction happens before the main-loop finally block.
        # Release any devices that were successfully opened before a required
        # peer failed, then propagate the startup error.
        stop_all_uart_readers()
        raise

    # Save the expanded source list and frame definitions to describe the streams
    # actually recorded.
    cfg["video_sources"] = expanded_video_configs
    cfg["numeric_sources"] = expanded_numeric_configs
    cfg["frame_definitions"] = build_frame_definitions(expanded_numeric_configs)

    recorders = list(numeric_recorders) + list(video_recorders)

    if not recorders:
        click.echo(
            f"{Colors.RED}Error: No data sources configured{Colors.RESET}", err=True
        )
        return

    # Construct RecorderManager before background workers: its initializer writes
    # the session config and Zarr metadata those workers require.
    recorder_manager = RecorderManager(
        recorders=recorders,
        record_path=data_path,
        config=cfg,
        verbose=is_verbose,
    )
    # Start the background aligner only after RecorderManager has written its config.
    attach_background_services(recorder_manager, cfg, data_path)

    visualizer = DataVisualizer(
        VisualizationConfig(
            enabled=viz_cfg.get("enabled", True),
            window_name=viz_cfg.get("window_name", "Data Recording"),
            resizable=viz_cfg.get("resizable", True),
            window_width=viz_cfg.get("window_width", None),
            window_height=viz_cfg.get("window_height", None),
            show_fps=viz_cfg.get("show_fps", True),
            show_episode_id=viz_cfg.get("show_episode_id", True),
            layout=viz_cfg.get("layout", "horizontal"),
            dashboard=viz_cfg.get("dashboard", False),
            overlay_numeric=viz_cfg.get("overlay_numeric", True),
            overlay_position=tuple(viz_cfg.get("overlay_position", [10, 60])),
            overlay_alpha=viz_cfg.get("overlay_alpha", 0.5),
            overlay_per_source_alpha=viz_cfg.get("overlay_per_source_alpha", None),
            text_scale=viz_cfg.get("text_scale", 1.0),
            tactile_components=viz_cfg.get("tactile_components", ["fz"]),
            tactile_text_scale=viz_cfg.get("tactile_text_scale", 1.5),
            values_per_line=viz_cfg.get("values_per_line", 8),
        )
    )

    # Keep the preview free of per-frame filesystem reads. Refresh this total
    # when finalization advances the episode ID (pending episodes do not count).
    saved_data_hours = None
    saved_episode_count = recorder_manager.get_saved_episode_count()
    saved_hours_episode_id = None

    # Get initial task description before starting streaming (streaming threads
    # can interfere with stdin input)
    meta = cfg.get("meta", {})
    initial_task = meta.get("task_description", cfg.get("task_description", None))
    if not initial_task:
        initial_task = input("Enter task description: ")
    recorder_manager.task_description = initial_task

    if not recorder_manager.start_streaming():
        click.echo("Failed to start streaming", err=True)
        # Release the UART readers that did open — otherwise a camera failing
        # after the exoskeletons connected leaves them claimed.
        stop_all_uart_readers()
        return
    # Bind each source to its opened device before reporting readiness. Refuse
    # the launch if any identity differs from the GUI identification scan.
    binding_refusal = _bind_launch_devices(
        recorder_manager, data_path, device_binding_scan
    )
    if binding_refusal:
        click.echo(f"{Colors.RED}{binding_refusal}{Colors.RESET}", err=True)
        _run_cleanup_steps(
            [
                ("stop_streaming", recorder_manager.stop_streaming),
                ("stop_all_uart_readers", stop_all_uart_readers),
            ]
        )
        raise SystemExit(DEVICE_BINDING_REFUSED_EXIT_CODE)
    click.echo(f"{Colors.BLUE}{Colors.BOLD}[Task]: {initial_task}{Colors.RESET}")

    click.echo("\n" + "=" * 50)
    click.echo("DATA RECORDING INSTRUCTIONS")
    click.echo("=" * 50)
    click.echo("  s - Start recording")
    click.echo("  w - Stop and save recording")
    click.echo("  a - Restart episode (discard)")
    click.echo("  t - Set task description (when not recording)")
    click.echo("  r - Acknowledge error and resume (when blocked by a red error)")
    click.echo("  q - Quit")
    click.echo("=" * 50 + "\n")

    task_input_mode = False
    task_input_buffer = ""

    # Set when a saved episode could not be finalized: its staging folder is
    # the only copy, so quit/shutdown must not discard it.
    keep_staging = False

    # Throttle USB gripper presence checks (full verify touches serial port enumeration).
    _gripper_check_interval_s = 0.5
    _last_gripper_check_m = 0.0

    # Blocking errors disable s/w/a/t until acknowledged with R. Each entry
    # contains a title, issues and a fix_hint.
    error_state: Optional[dict] = None

    # Active frame-drop / frame-delay warnings polled from recorders each iteration.
    # Non-blocking: all recording controls remain functional while warnings are shown.
    current_warnings: dict = {}
    prev_warning_keys: set = set()

    viz_fps = viz_cfg.get("fps", 30)
    rate_keeper = RateKeeper(target_rate=viz_fps, name="MainLoop")

    # Set RECORD_TIMING=1 to measure frame retrieval, mosaic/overlay drawing and
    # OpenCV event processing when diagnosing low preview frame rates.
    _timing_on = os.environ.get("RECORD_TIMING", "") not in ("", "0", "false")
    _t_acc = {"frames": 0.0, "combine": 0.0, "overlay": 0.0, "show": 0.0, "loop": 0.0}
    _t_iters = 0
    _t_report_every = max(1, int(viz_fps))  # ~once per second
    _t_loop_start = time.perf_counter()

    try:
        while True:
            with rate_keeper:
                if _timing_on:
                    _now = time.perf_counter()
                    _t_acc["loop"] += _now - _t_loop_start
                    _t_loop_start = _now
                    _t_iters += 1
                    if _t_iters >= _t_report_every:
                        n = _t_iters
                        avg = {k: 1e3 * v / n for k, v in _t_acc.items()}
                        work = (
                            avg["frames"]
                            + avg["combine"]
                            + avg["overlay"]
                            + avg["show"]
                        )
                        loop_fps = 1e3 / avg["loop"] if avg["loop"] > 0 else 0.0
                        print(
                            f"[timing] loop {loop_fps:5.1f} fps | "
                            f"work {work:5.1f}ms = "
                            f"frames {avg['frames']:4.1f} + "
                            f"combine {avg['combine']:4.1f} + "
                            f"overlay {avg['overlay']:4.1f} + "
                            f"show {avg['show']:4.1f} "
                            f"(target {viz_fps} fps = {1e3 / viz_fps:.1f}ms)"
                        )
                        _t_acc = {k: 0.0 for k in _t_acc}
                        _t_iters = 0

                _t0 = time.perf_counter()
                frames = recorder_manager.get_latest_frames()
                if _timing_on:
                    _t_acc["frames"] += time.perf_counter() - _t0

                if recorder_manager.is_recording:
                    now_m = time.monotonic()
                    if now_m - _last_gripper_check_m >= _gripper_check_interval_s:
                        _last_gripper_check_m = now_m
                        assert_all_uart_readers_connected()

                # Separate video and numeric frames (order matches combine_frames)
                video_frames = []
                video_source_order: list[str] = []
                numeric_data = {}
                pose_data = {}

                for name, frame_list in frames.items():
                    if frame_list and frame_list[0] is not None:
                        frame = frame_list[0]
                        if hasattr(frame, "rgb") and frame.rgb is not None:
                            video_frames.append(frame.rgb)
                            video_source_order.append(name)
                            if hasattr(frame, "pose") and frame.pose is not None:
                                pose_data[name] = frame.pose
                        elif isinstance(frame, NumericFrame):
                            numeric_data[name] = frame_list

                # Live stream warnings. The dashboard reserves warning space;
                # the alternate preview uses an overlay.
                current_warnings = recorder_manager.get_active_warnings()
                new_warning_keys = set(current_warnings.keys()) - prev_warning_keys
                if new_warning_keys and not visualizer.config.dashboard:
                    visualizer.trigger_shake(duration=0.8, intensity=12)
                prev_warning_keys = set(current_warnings.keys())

                # Discard the take on device disconnect (source_lost).
                lost_sources = [
                    (src, msg)
                    for (src, wtype), msg in current_warnings.items()
                    if wtype == "source_lost"
                ]
                if lost_sources and error_state is None:
                    lost_names = [src for src, _ in lost_sources]
                    issues = [
                        f"Device disconnected: {src} -- {msg}"
                        for src, msg in lost_sources
                    ]
                    if recorder_manager.is_recording:
                        visualizer.stop_timer()
                        recorder_manager.stop_recording()
                        recorder_manager.clear_recording()
                        click.echo(
                            f"{Colors.RED}[!] Device lost: {', '.join(lost_names)} — episode discarded{Colors.RESET}"
                        )
                    else:
                        click.echo(
                            f"{Colors.RED}[!] Device lost: {', '.join(lost_names)}{Colors.RESET}"
                        )
                    error_state = {
                        "title": "DEVICE DISCONNECTED -- episode discarded",
                        "issues": issues,
                        "fix_hint": "Restart session to reconnect -- press Q",
                        "fatal": True,
                    }
                    visualizer.trigger_shake()

                display_revision = (recorder_manager.episode_id, saved_episode_count)
                if (
                    saved_episode_count is not None
                    and saved_hours_episode_id != display_revision
                ):
                    saved_data_hours = recorder_manager.get_total_data_hours()
                    saved_episode_count = recorder_manager.get_saved_episode_count()
                    saved_hours_episode_id = (
                        recorder_manager.episode_id,
                        saved_episode_count,
                    )

                if visualizer.config.enabled:
                    _t0 = time.perf_counter()
                    combined = (
                        visualizer.combine_frames(
                            video_frames, labels=video_source_order
                        )
                        if video_frames
                        else np.zeros((400, 640, 3), dtype=np.uint8)
                    )
                    if _timing_on:
                        _t_acc["combine"] += time.perf_counter() - _t0

                    _t0 = time.perf_counter()
                    # The dashboard owns status/tactile text outside the camera mosaic.
                    if not visualizer.config.dashboard:
                        combined = visualizer.draw_overlay(
                            combined,
                            episode_id=recorder_manager.episode_id,
                            is_recording=recorder_manager.is_recording,
                            numeric_data=numeric_data if numeric_recorders else None,
                            pose_data=pose_data if pose_data else None,
                            error_active=error_state is not None,
                            saved_episode_count=saved_episode_count,
                            show_saved_count=True,
                        )

                    if visualizer.config.dashboard:
                        status_text = "Idle"
                        start_hint = "S  Start recording"
                        if recorder_manager.is_recording:
                            status_text = "Recording"
                            start_hint = "W  Stop and save"
                        if error_state is not None:
                            status_text = "Needs attention"
                        combined = visualizer.draw_dashboard(
                            combined,
                            episode_id=recorder_manager.episode_id,
                            is_recording=recorder_manager.is_recording,
                            numeric_data=numeric_data,
                            task_description=recorder_manager.task_description,
                            error_active=error_state is not None,
                            status_text=status_text,
                            start_hint=start_hint,
                            saved_data_hours=saved_data_hours,
                            saved_episode_count=saved_episode_count,
                            show_saved_count=True,
                            warnings=current_warnings,
                        )
                    else:
                        combined = visualizer.draw_task_display(
                            combined, recorder_manager.task_description
                        )

                    if task_input_mode:
                        combined = visualizer.draw_text_input_overlay(
                            combined,
                            prompt="Task Description:",
                            current_text=task_input_buffer,
                        )

                    # Apply any active shake to the base frame BEFORE drawing
                    # warning / error overlays. No-op when no shake is active.
                    # Used by error states and frame-drop warnings.
                    combined = visualizer.apply_shake(combined)

                    # The dashboard reserves warning space; only the alternate preview
                    # needs an overlay.
                    if (
                        not visualizer.config.dashboard
                        and current_warnings
                        and error_state is None
                    ):
                        combined = visualizer.draw_warning_overlay(
                            combined, current_warnings
                        )

                    if _timing_on:
                        _t_acc["overlay"] += time.perf_counter() - _t0

                    _t0 = time.perf_counter()
                    # Draw blocking errors last so nothing obscures their
                    # details or the operator's action hint.
                    if error_state is not None:
                        combined = visualizer.draw_error_overlay(
                            combined,
                            title=error_state["title"],
                            issues=error_state["issues"],
                            fix_hint=error_state["fix_hint"],
                            page=error_state.get("page", 0),
                        )
                    key = visualizer.show(combined)
                    if _timing_on:
                        _t_acc["show"] += time.perf_counter() - _t0
                else:
                    # No video, just poll for keys
                    key = cv2.waitKey(1) & 0xFF

                if task_input_mode:
                    if key == 27:  # Escape - cancel
                        task_input_mode = False
                        task_input_buffer = ""
                        click.echo("Task input cancelled")
                    elif key == 13:  # Enter - confirm
                        recorder_manager.task_description = task_input_buffer
                        click.echo(
                            f"{Colors.BLUE}{Colors.BOLD}[Task]: {task_input_buffer}{Colors.RESET}"
                        )
                        task_input_mode = False
                        task_input_buffer = ""
                    elif key == 8 or key == 127:  # Backspace
                        task_input_buffer = task_input_buffer[:-1]
                    elif 32 <= key <= 126:  # Printable ASCII characters
                        task_input_buffer += chr(key)
                    continue  # Skip other key handling while in input mode

                # Blocking errors allow issue paging, R (acknowledge), and Q (quit).
                if error_state is not None and key != ord("q"):
                    if key in (ord("["), ord("]")):
                        error_state["page"] = error_state.get("page", 0) + (
                            1 if key == ord("]") else -1
                        )
                    elif key == ord("r") and not error_state.get("fatal", False):
                        error_state = None
                        click.echo(
                            f"{Colors.BLUE}Error acknowledged -- ready to record.{Colors.RESET}"
                        )
                    continue

                if key == ord("q"):
                    click.echo("\nQuitting...")
                    if recorder_manager.is_recording:
                        recorder_manager.stop_recording()
                        recorder_manager.clear_recording()
                    if keep_staging:
                        click.echo(
                            f"{Colors.YELLOW}Keeping unfinalized staging "
                            f"{recorder_manager.get_pending_episode_path()} for "
                            f"recovery{Colors.RESET}"
                        )
                    else:
                        recorder_manager.discard_staging()
                    break

                elif key == ord("t"):
                    if not recorder_manager.is_recording:
                        task_input_mode = True
                        task_input_buffer = ""
                        click.echo("Entering task input mode (type in video window)")
                    else:
                        click.echo("Cannot change task while recording")

                elif key == ord("s"):
                    if keep_staging and os.path.isdir(
                        recorder_manager.get_pending_episode_path()
                    ):
                        error_state = _save_error_state(
                            [
                                f"{recorder_manager.get_pending_episode_path()} "
                                "kept on disk for recovery"
                            ],
                            keep_staging=True,
                        )
                        click.echo(error_state["fix_hint"])
                    elif recorder_manager.is_recording:
                        click.echo("Already recording")
                    else:
                        visualizer.start_timer()
                        visualizer.reset_fps_counter()
                        if recorder_manager.reset_episode_recording():
                            # A new take replaces whatever staging was kept.
                            keep_staging = False
                            recorder_manager.start_recording()
                            click.echo(
                                f"{Colors.BLUE}{Colors.BOLD}Recording started - Episode {recorder_manager.episode_id}{Colors.RESET}"
                            )

                elif key == ord("w"):
                    if recorder_manager.is_recording:
                        visualizer.stop_timer()
                        click.echo("Saving recording to staging...")
                        if recorder_manager.stop_recording():
                            # Write to the pending folder (episode_N.pending/),
                            # then rename it to episode_N.
                            recorder_manager.save_recordings(pending=True)
                            saved_episode_id = recorder_manager.episode_id
                            issues = _finalize_episode(
                                recorder_manager, saved_episode_id
                            )
                            if issues is None:
                                click.echo(
                                    f"{Colors.GREEN}[+] Episode {saved_episode_id} saved{Colors.RESET}"
                                )
                                visualizer.flash_timer_saved()
                                total_hours = recorder_manager.get_total_data_hours()
                                click.echo(
                                    f"{Colors.PURPLE}Total data collected: {total_hours:.2f} hours{Colors.RESET}"
                                )
                            else:
                                click.echo(
                                    f"{Colors.RED}Warning: Failed to finalize episode{Colors.RESET}"
                                )
                                keep_staging = os.path.isdir(
                                    recorder_manager.get_pending_episode_path()
                                )
                                error_state = _save_error_state(
                                    issues, keep_staging=keep_staging
                                )
                                visualizer.trigger_shake()
                    else:
                        click.echo("Not recording")

                elif key == ord("a"):
                    if recorder_manager.is_recording:
                        visualizer.stop_timer()
                        click.echo(
                            f"{Colors.BLUE}{Colors.BOLD}Restarting episode...{Colors.RESET}"
                        )
                        recorder_manager.stop_recording()
                        recorder_manager.clear_recording()
                        visualizer.reset_fps_counter()
                    else:
                        click.echo(
                            f"{Colors.BLUE}{Colors.BOLD}Not recording -- nothing to discard{Colors.RESET}"
                        )

    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")

    finally:
        # Every step runs even if an earlier one raises (e.g. a video writer
        # release error from discard_staging): device teardown and the
        # background-service drain must not be skipped.
        _run_cleanup_steps(
            [
                ("stop_recording", recorder_manager.stop_recording),
                ("stop_streaming", recorder_manager.stop_streaming),
                (
                    "discard_staging",
                    lambda: (
                        None if keep_staging else recorder_manager.discard_staging()
                    ),
                ),
                ("visualizer.close", visualizer.close),
                ("stop_all_uart_readers", stop_all_uart_readers),
                # Drain pending alignments before exit.
                (
                    "shutdown_background_services",
                    recorder_manager.shutdown_background_services,
                ),
            ]
        )
        click.echo("Recording stopped")


if __name__ == "__main__":
    main()
