# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Composable Data Recording Script

The recorder calibrates the cameras inline when a calibration is due.

Usage:
    uv run real_scripts/data_collection/record_data.py --config config.yaml

Controls:
    s - First press: take the start pose; second press: run pre-episode
        checks and start recording if they pass
    w - Stop and save recording
    a - Restart episode (discard current)
    g - Re-roll this episode's placement layout
    r - Acknowledge an error
    q - Quit
"""

import hashlib
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from enum import Enum
from pathlib import Path
from typing import Optional

import click
import cv2

from real_scripts.data_collection.pre_episode_checks.camera_calibration import (
    CameraRelativeCalibrationCheck,
)
from real_scripts.data_processing.data_check import (
    check_dataset,
    data_check_findings,
    overlay_hint,
    overlay_lines,
    terminal_lines,
    warning_lines,
)
from umi.common.data.frame import NumericFrame, TrackingFrame
from umi.common.recording import shutdown_progress
from umi.common.recording.base_recorder import PENDING_SUFFIX
from umi.common.recording.pre_episode_checks import (
    PreEpisodeContext,
    build_checks,
    run_pre_episode_checks,
)
from umi.common.recording.record_manager import RecorderManager
from umi.common.recording.session_slot import (
    resolve_session_slot,
    save_launch_config,
)
from umi.common.tasks import registry_gap
from umi.common.utility import setup_variants
from umi.common.utility.background_aligner import attach_background_services
from umi.common.utility.data_visualizer import build_visualizer
from umi.common.utility.log_capture import tee_stdio_to_file, worker_redirect_stdio
from umi.common.utility.output_layout import default_session_name, session_dir
from umi.common.utility.print_utils import Colors
from umi.common.utility.rate_keeper import RateKeeper
from umi.common.utility.recording_utils import (
    assert_all_uart_readers_connected,
    build_frame_definitions,
    create_numeric_recorders,
    create_video_recorders,
    load_config,
    release_constructed_devices,
    stop_all_oak_devices,
    stop_all_uart_readers,
)
from umi.common.utility.signal_handling import install_stop_on_sigterm


class RecordingStage(Enum):
    """Operator's progress through the pre-record sequence.

    IDLE            -> 1st `s`: starts inline calibration (if due) or
                       moves on to INIT_POSE. No
                       pre-episode gate runs here -- it runs at the
                       recording-start press below.
    CALIBRATION     -> inline ArUco/ChArUco calibration in progress; tick() is
                       called every frame until convergence or timeout.
    INIT_POSE       -> the operator takes the start pose; the next `s` runs the
                       pre-episode gate (gripper UART connected, ...) and
                       starts recording only if it passes. A failed
                       gate keeps the operator in INIT_POSE so they can adjust
                       and retry.
    """

    IDLE = "idle"
    CALIBRATION = "calibration"
    INIT_POSE = "init_pose"


# =============================================================================
# Main Recording Logic
# =============================================================================


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(),
    default=None,
    help="Session folder to record into. Overrides recording.data_dir from the "
    "config; when neither is set the session goes to "
    "output/raw/{task}/{date}_{task} (see umi.common.utility.output_layout).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--check-config",
    type=click.Path(exists=True),
    default="config/data_check/data_check_config.yaml",
    help="Path to data check config YAML file for quality validation",
)
@click.option(
    "--ignore-data-check",
    is_flag=True,
    default=False,
    help="Ignore data check",
)
@click.option(
    "--ignore-pre-episode-check",
    is_flag=True,
    default=False,
    help="Skip pre-episode checks (gripper link, calibration) before recording",
)
@click.option(
    "--allow-missing-devices",
    is_flag=True,
    default=False,
    help="Record even if a declared camera or gripper cannot be opened — the "
    "session will be missing that device and the data check cannot detect it.",
)
def main(
    config: str,
    data_dir: Optional[str],
    verbose: bool,
    check_config: Optional[str],
    ignore_data_check: bool,
    ignore_pre_episode_check: bool,
    allow_missing_devices: bool,
):
    """Composable data recording with configurable video and numeric sources."""
    # Before anything can block: a supervisor that cannot reach the cv2 window
    # must still be able to stop this process without losing the alignment
    # drain.
    install_stop_on_sigterm()

    cfg = load_config(config)
    recording_cfg = cfg.setdefault("recording", {})
    viz_cfg = cfg.get("visualization", {})

    if data_dir:
        recording_cfg["data_dir"] = data_dir
    if not recording_cfg.get("data_dir"):
        task_name = (cfg.get("meta") or {}).get("task_name") or ""
        recording_cfg["data_dir"] = session_dir(
            "raw", task_name, default_session_name(task_name)
        )
    if verbose:
        recording_cfg["verbose"] = True

    # Roll to a new session UID if acquisition settings or calibration conflict
    # with the current folder. Save the original path before cfg is updated so
    # the rollover can be logged after session logging starts.
    orig_data_path = os.path.expanduser(
        str(recording_cfg.get("data_dir", "./data")).rstrip("/")
    )
    data_path = resolve_session_slot(cfg, check_config, os.path.expanduser(config))
    os.makedirs(data_path, exist_ok=True)
    # Snapshot the unexpanded launch config for session conflict detection
    # before _run_session adds per-source defaults.
    save_launch_config(data_path, cfg)

    # Append session output to record_data.log with a divider for each launch.
    log_path = os.path.join(data_path, "record_data.log")
    status = 0
    try:
        with tee_stdio_to_file(log_path):
            if data_path != orig_data_path:
                click.echo(
                    f"{Colors.YELLOW}Config changed vs the session already in "
                    f"{orig_data_path}; rolled onto a new session and recording "
                    f"into: {data_path}{Colors.RESET}"
                )
            _run_session(
                cfg,
                recording_cfg,
                viz_cfg,
                data_path,
                check_config,
                ignore_data_check,
                ignore_pre_episode_check,
                allow_missing_devices=allow_missing_devices,
            )
    except SystemExit as exc:
        # A specific exit status was requested; keep it rather than reducing it
        # to the 0/1 below.
        status = (
            exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        )
    except BaseException:
        # The session finally block already logged the exception while output was
        # redirected.
        status = 1
    finally:
        # Native camera SDK threads can outlive device.close() and block interpreter
        # shutdown. Exit only after hardware cleanup and the session-log context
        # have finished. Flush output explicitly and preserve the session exit
        # status because os._exit skips Python cleanup and buffered I/O.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)


def _relaxed_pre_checks(cfg: dict) -> list[str]:
    return [
        name
        for name, enforce in (
            ((cfg.get("check_enforcement") or {}).get("pre")) or {}
        ).items()
        if enforce is False
    ]


def _calibration_due(check, episode_id: int) -> bool:
    """Warn-only calibration readiness never holds the recording stage."""
    return (
        check is not None
        and getattr(check, "enforce", True)
        and check.is_due(episode_id)
    )


def _tick_calibration_stage(
    calibration_check: CameraRelativeCalibrationCheck,
    ctx,
    visualizer,
) -> RecordingStage:
    """One frame of the CALIBRATION stage. Returns the stage to be in next.

    Convergence is checked before the timeout: `_converged` latches, and a
    calibration that converged on its last frame must report that it finished
    rather than that it gave up.
    """
    was_converged = calibration_check.is_session_converged()
    calibration_check.tick(ctx)

    if calibration_check.is_session_converged():
        if not was_converged:
            click.echo(
                f"{Colors.GREEN}[calibration] Converged — "
                f"press s to continue.{Colors.RESET}"
            )
        # The DONE overlay keeps rendering until the operator presses `s`, so
        # the stage does not advance on its own.
        return RecordingStage.CALIBRATION

    if calibration_check.is_session_timed_out():
        click.echo(
            f"{Colors.YELLOW}[calibration] Timed out — "
            f"hold marker visible to all cameras and press s to retry."
            f"{Colors.RESET}"
        )
        visualizer.trigger_shake(duration=0.8, intensity=12)
        calibration_check.reset_session()
        return RecordingStage.IDLE

    return RecordingStage.CALIBRATION


def _enter_init_pose() -> RecordingStage:
    """Enter INIT_POSE: the way into it from idle with no calibration due and
    from an acknowledged calibration DONE."""
    click.echo(
        f"{Colors.BLUE}[Stage] INIT_POSE -- "
        f"adjust to the start pose, press s again to run the "
        f"pre-episode checks and start recording.{Colors.RESET}"
    )
    return RecordingStage.INIT_POSE


# Bound the wait for a pending data-check verdict below the console's stall
# limit. If it expires, quarantine the episode as unverified. Emit periodic
# progress so the console can distinguish waiting from a stalled recorder.
SHUTDOWN_CHECK_WAIT_S = 120.0

# Emit shutdown progress periodically while waiting for a data-check verdict.
SHUTDOWN_CHECK_HEARTBEAT_S = 15.0


def _instruction_panel_spec(instruction, task_description):
    # Use consistent colors for setup and task headings across preview windows.
    text = f"SETUP\n{instruction}"
    if task_description:
        text += f"\n\nTASK\n{task_description}"
    return {
        "text": text,
        "title": "Operator Instruction",
        "header_lines": ["SETUP", "TASK"],
        "header_color": (100, 200, 255),
    }


def _setup_panel(setup, task_description, warning):
    """Cell 0: the episode's resolved setup, registry warning above it."""
    return registry_gap.panel_with_warning(
        _instruction_panel_spec(setup, task_description), warning
    )


def _settle_pending_check(
    recorder_manager: RecorderManager,
    pending_check_future,
    episode_id,
) -> None:
    """Give a `w`-saved episode its verdict before shutdown discards staging.

    `w` does not finalize. It writes `episode_N.pending/` and hands the data
    check to a subprocess; only when that check comes back PASSED does the main
    loop rename the folder to `episode_N/`. So between pressing `w` and the
    check finishing (seconds to minutes on a large episode), a saved episode
    exists only as a pending folder.

    `_run_session`'s `finally` calls `discard_staging()` unconditionally, and
    that is `shutil.rmtree`. Without this step, pressing Stop in that window
    (for example right after saving the last episode of the day) would delete
    an episode the console had already reported as saved, and abandon the
    check that would have kept it.

    Resolving the check here moves the folder out of staging (finalize renames
    it, quarantine moves it), so the `discard_staging()` that follows finds
    nothing and only ever aborts an episode that was never saved.

    Never deletes. The in-loop handler discards on a check ERROR because an
    operator is watching a red banner and can act on it; nobody is watching
    here, and quarantine is recoverable where rmtree is not. An unverified
    episode in quarantine is a review task, the right cost for a missing
    verdict.
    """
    if pending_check_future is None:
        return

    click.echo(
        f"{Colors.BLUE}Data check for episode {episode_id} still running; "
        f"waiting up to {SHUTDOWN_CHECK_WAIT_S:.0f}s for its verdict before "
        f"shutting down.{Colors.RESET}"
    )
    deadline = time.monotonic() + SHUTDOWN_CHECK_WAIT_S
    report = None
    failure = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure = f"data check did not finish within {SHUTDOWN_CHECK_WAIT_S:.0f}s"
            break
        try:
            report = pending_check_future.result(
                timeout=min(SHUTDOWN_CHECK_HEARTBEAT_S, remaining)
            )
            break
        except FuturesTimeoutError:
            click.echo(f"  still checking episode {episode_id}...")
        except Exception as exc:  # noqa: BLE001 - any check failure ends the wait
            failure = f"data check errored: {exc}"
            break

    issues = None
    findings = None
    if failure is None:
        episode_key = f"episode_{episode_id}{PENDING_SUFFIX}"
        try:
            result = report["episodes"][episode_key]
            findings = data_check_findings(result)
            for line in warning_lines(result.get("warnings", [])):
                click.echo(line)
            if result["passed"]:
                if recorder_manager.finalize_episode(findings=findings):
                    click.echo(
                        f"{Colors.GREEN}[+] Data check PASSED - episode "
                        f"{episode_id} finalized during shutdown{Colors.RESET}"
                    )
                    return
                # Finalize failed with the folder still in staging. Fall through
                # to quarantine rather than leave it for `discard_staging`.
                failure = "episode passed its data check but could not be finalized"
            else:
                issues = list(result["issues"])
        except (KeyError, TypeError) as exc:
            failure = f"data check returned no verdict for this episode: {exc}"

    if issues is None:
        issues = [failure]
    if recorder_manager.quarantine_staging(issues=issues, findings=findings):
        click.echo(
            f"{Colors.YELLOW}[!] Episode {episode_id} quarantined during "
            f"shutdown:{Colors.RESET}"
        )
        for line in terminal_lines(issues):
            click.echo(line)
    else:
        click.echo(
            f"{Colors.RED}Warning: could not quarantine episode {episode_id} "
            f"during shutdown. Manual cleanup may be required.{Colors.RESET}"
        )


def _run_session(
    cfg: dict,
    recording_cfg: dict,
    viz_cfg: dict,
    data_path: str,
    check_config: Optional[str],
    ignore_data_check: bool,
    ignore_pre_episode_check: bool,
    allow_missing_devices: bool,
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

    # Video recorder construction cleans up its own partial failures; avoid
    # double-closing.
    video_recorders, video_sources, expanded_video_configs = create_video_recorders(
        video_source_configs=cfg.get("video_sources", []),
        record_path=data_path,
        default_record_fps=default_record_fps,
        default_stream_fps=default_stream_fps,
        default_frame_type=default_video_frame_type,
        verbose=is_verbose,
        info_verbose=True,  # Always show video recorder info
        warning_verbose=True,  # Always show video recorder warnings
        video_encoder=video_encoder,
        allow_missing_devices=allow_missing_devices,
        require_all_sources=require_all_sources,
    )

    try:
        numeric_recorders, _, expanded_numeric_configs = create_numeric_recorders(
            numeric_source_configs=cfg.get("numeric_sources", []),
            record_path=data_path,
            default_record_fps=default_record_fps,
            default_stream_fps=default_stream_fps,
            default_frame_type=default_numeric_frame_type,
            verbose=is_verbose,
            info_verbose=True,
            warning_verbose=True,
            recording_meta=cfg.get("meta"),
            allow_missing_devices=allow_missing_devices,
            require_all_sources=require_all_sources,
        )
    except Exception:
        # Numeric-source construction occurs before the main cleanup block. On
        # failure, release already-open cameras and drain shared OAK/UART caches.
        release_constructed_devices(video_sources)
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
        data_check_config_path=check_config,
        verbose=is_verbose,
    )

    # Discover per-episode checks from pre_episode_checks/*.py.
    # --ignore-pre-episode-check bypasses calibration and check overlays but
    # retains the idle → init-pose → recording interaction.
    if ignore_pre_episode_check:
        pre_episode_checks = []
        click.echo(f"{Colors.YELLOW}Ignoring pre-episode checks{Colors.RESET}")
    else:
        pre_episode_checks = build_checks(
            cfg.get("pre_episode_checks"),
            search_dirs=[Path(__file__).parent / "pre_episode_checks"],
            relaxed=_relaxed_pre_checks(cfg),
        )
    if pre_episode_checks:
        names = ", ".join(c.name for c in pre_episode_checks)
        click.echo(f"{Colors.BLUE}Pre-episode checks: {names}{Colors.RESET}")
        relaxed = [
            c.name for c in pre_episode_checks if not getattr(c, "enforce", True)
        ]
        if relaxed:
            click.echo(
                f"{Colors.YELLOW}Warn-only pre-episode checks: {', '.join(relaxed)}{Colors.RESET}"
            )

    # Locate the calibration check for refresh gating.
    calibration_check: Optional[CameraRelativeCalibrationCheck] = next(
        (c for c in pre_episode_checks if c.name == "camera.relative_calibration"),
        None,
    )

    # Attach background alignment only after RecorderManager has written the
    # session config. Enabled alignment requires meta.task_name.
    attach_background_services(recorder_manager, cfg, data_path)

    # Task-registry instructions and the per-session 5x7 object-placement grid
    # precede configured filler cells. Registry failures must not block recording.
    def _layout_seed(session_uid, episode_id):
        # Stable per-episode seed for the placement grid: a discarded or
        # quarantined attempt keeps its episode id (see RecorderManager
        # .discard_staging / .quarantine_staging), so the operator re-stages
        # the identical scene; finalize_episode advances the id and with it
        # the layout. sha256, not hash(): Python's hash is salted per process.
        key = f"{session_uid}:{episode_id}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")

    task_name = cfg.get("meta", {}).get("task_name")
    operator_instruction = ""  # raw registry text, setup-variant groups kept
    warning = None
    scene_objects = None
    layout_session_uid = cfg.get("meta", {}).get("session_uid") or data_path
    placement_episode_id = None  # episode the shown layout was rendered for
    layout_reroll = 0  # operator 'g' re-rolls within the episode; reset on advance
    setup_episode_id = None  # episode the shown setup variant was resolved for

    def current_setup():
        # This episode's options for any {a|b} groups; per episode id only,
        # so 'g' re-rolls and discards leave it alone.
        return setup_variants.resolve(
            operator_instruction, layout_session_uid, recorder_manager.episode_id
        )

    from umi.common.utility.placement_grid import (
        render_layout,
        render_placement_grid,
        sample_layout,
    )

    # Both panels always exist, so tasks with no operator_instruction / objects
    # in the registry still work: an empty instruction panel and a bare lattice
    # with no objects marked.
    extras = [
        {"text": "", "title": "Operator Instruction"},
        {
            "array": render_placement_grid("", empty_ok=True),
            "title": "Placement Grid",
        },
    ]
    if task_name:
        try:
            spec, registry_info, refetched = registry_gap.resolve_task_spec(
                task_name,
                log=lambda line: click.echo(f"{Colors.YELLOW}{line}{Colors.RESET}"),
            )
            recorder_manager.registry_provenance = registry_gap.registry_provenance(
                spec, registry_info, refetched
            )
            warning = registry_gap.registry_warning(task_name, spec, registry_info)
            if warning:
                click.echo(
                    f"{Colors.RED}{Colors.BOLD}[Registry] {warning}{Colors.RESET}"
                )
            operator_instruction = spec.operator_instruction if spec else ""
            # One panel: the instruction when there is one, and the registry
            # warning above it (or alone) when there is one. The terminal
            # scrolls; the panel stays for the whole session (re-rendered per
            # episode only when the setup has {a|b} groups).
            panel = (
                _instruction_panel_spec(current_setup(), spec.short_description)
                if operator_instruction
                else None
            )
            if warning or panel:
                extras[0] = registry_gap.panel_with_warning(panel, warning)
            if operator_instruction:
                # Stamped onto every saved episode's zarr attrs alongside the
                # placement layout (RecorderManager._save_scene_provenance).
                recorder_manager.operator_instruction = current_setup()
                if setup_variants.has_variants(operator_instruction):
                    setup_episode_id = recorder_manager.episode_id
                # Structured scene objects (with containment) beat prose parsing.
                scene_objects = spec.objects or None
                layout = sample_layout(
                    operator_instruction,
                    objects=scene_objects,
                    seed=_layout_seed(layout_session_uid, recorder_manager.episode_id),
                )
                if layout is not None:
                    extras[1] = {
                        "array": render_layout(layout),
                        "title": "Placement Grid",
                    }
                    placement_episode_id = recorder_manager.episode_id
                    recorder_manager.placement_layout = layout
                click.echo(
                    "Operator instruction loaded from task registry"
                    + (" (with placement grid)." if layout is not None else ".")
                )
            elif not warning:
                # The registry warning above already said why the panel is
                # empty; this line is for the quiet case only.
                click.echo(
                    "No operator instruction for this task; showing empty "
                    "instruction panel and bare placement grid."
                )
        except Exception as e:
            click.echo(
                f"{Colors.YELLOW}Task registry unavailable; empty operator "
                f"instruction panel: {e}{Colors.RESET}"
            )
    viz_cfg["filler_cells"] = extras + list(viz_cfg.get("filler_cells") or [])
    # Without panel content, a side column / extra windows would just be an
    # empty half-window; degrade to spare-cell mode so cameras keep the space.
    if not operator_instruction and viz_cfg.get("filler_position") in (
        "side",
        "windows",
    ):
        viz_cfg["filler_position"] = "cells"
        click.echo("No panel content; using spare grid cells for the panels.")

    visualizer = build_visualizer(viz_cfg)
    if placement_episode_id is not None:
        visualizer.layout_hint = "press 'g' to change layout"

    # Get initial task description before starting streaming (streaming threads
    # can interfere with stdin input)
    meta = cfg.get("meta", {})
    initial_task = meta.get("task_description", cfg.get("task_description", None))
    if not initial_task:
        initial_task = input("Enter task description: ")
    recorder_manager.task_description = initial_task
    # This is the description every episode of the session records; there is
    # no way to change it from the keyboard once streaming starts.
    if operator_instruction:
        visualizer.update_filler_cell(
            0, _setup_panel(current_setup(), initial_task, warning)
        )

    if not recorder_manager.start_streaming():
        click.echo("Failed to start streaming", err=True)
        return
    click.echo(f"{Colors.BLUE}{Colors.BOLD}[Task]: {initial_task}{Colors.RESET}")

    click.echo("\n" + "=" * 50)
    click.echo("DATA RECORDING INSTRUCTIONS")
    click.echo("=" * 50)
    click.echo("  s - Take the start pose, then s again to start recording")
    click.echo("  w - Stop and save recording")
    click.echo("  a - Restart episode (discard)")
    click.echo("  g - Re-roll the placement grid layout (when not recording)")
    click.echo("  r - Acknowledge error and resume (when blocked by a red error)")
    click.echo("  q - Quit")
    click.echo("=" * 50 + "\n")

    # Warn the operator if inline calibration will be required before the first episode.
    if _calibration_due(calibration_check, recorder_manager.episode_id):
        click.echo(
            f"{Colors.YELLOW}[calibration] Inline camera calibration is required "
            f"before recording.\n"
            f"  Hold the ArUco/ChArUco marker visible to all calibration cameras,\n"
            f"  then press s to start calibration. Recording will unlock once the\n"
            f"  calibration converges (overlay shows DONE).{Colors.RESET}\n"
        )

    # Run data checks in a subprocess to avoid starving camera threads on the GIL.
    # Forward worker output to the session log.
    check_executor = ProcessPoolExecutor(
        max_workers=1, initializer=worker_redirect_stdio
    )
    pending_check_future = None
    pending_check_episode_id = None

    stage: RecordingStage = RecordingStage.IDLE

    # Throttle USB gripper presence checks (full verify touches serial port enumeration).
    _gripper_check_interval_s = 0.5
    _last_gripper_check_m = 0.0

    # Blocking error state: populated when a red-level error requires the operator
    # to acknowledge before recording can resume.  Keys s/w/a are blocked while
    # this is set; pressing R clears it.
    # Format: {"title": str, "issues": List[str], "fix_hint": str}
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

    def _advance_past_idle() -> RecordingStage:
        """Transition out of IDLE into the calibration stage if inline
        calibration is due, otherwise into INIT_POSE. Returns the new stage.
        """
        episode_id = recorder_manager.episode_id
        if _calibration_due(calibration_check, episode_id):
            click.echo(
                f"{Colors.BLUE}[calibration] Starting inline calibration — "
                f"hold the marker visible to all cameras.{Colors.RESET}"
            )
            calibration_check.start_session()
            return RecordingStage.CALIBRATION
        # No calibration needed: advance to INIT_POSE. The pre-episode gate
        # runs at the recording-start press (INIT_POSE) below, so it validates
        # the state the instant recording begins.
        return _enter_init_pose()

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

                # ------- per-episode setup variant -------
                # Own tracker, not the layout's: a task with {a|b} groups may
                # have no placement layout, and its setup must still advance.
                if (
                    setup_episode_id is not None
                    and recorder_manager.episode_id != setup_episode_id
                ):
                    setup_episode_id = recorder_manager.episode_id
                    setup = current_setup()
                    recorder_manager.operator_instruction = setup
                    try:
                        visualizer.update_filler_cell(
                            0,
                            _setup_panel(
                                setup, recorder_manager.task_description, warning
                            ),
                        )
                    except Exception as e:  # preview must never stop recording
                        click.echo(
                            f"{Colors.YELLOW}Setup panel re-render failed: "
                            f"{e}{Colors.RESET}"
                        )
                    click.echo(f"Setup for episode {setup_episode_id}: {setup}")

                # ------- per-episode placement layout -------
                # Seeded by (session_uid, episode_id): finalize advances the
                # id -> a fresh layout; discard/quarantine keep the id -> the
                # operator re-stages the identical scene. Checked per tick so
                # every finalize path (data check, --ignore-data-check,
                # shutdown finalize) is covered by the same line.
                if (
                    placement_episode_id is not None
                    and recorder_manager.episode_id != placement_episode_id
                ):
                    placement_episode_id = recorder_manager.episode_id
                    layout_reroll = 0
                    try:
                        new_layout = sample_layout(
                            operator_instruction,
                            objects=scene_objects,
                            seed=_layout_seed(layout_session_uid, placement_episode_id),
                        )
                        if new_layout is not None:
                            recorder_manager.placement_layout = new_layout
                            visualizer.update_filler_cell(
                                1,
                                {
                                    "array": render_layout(new_layout),
                                    "title": "Placement Grid",
                                },
                            )
                            click.echo(
                                f"Placement layout re-rolled for episode "
                                f"{placement_episode_id}"
                            )
                    except Exception as e:  # preview must never stop recording
                        click.echo(
                            f"{Colors.YELLOW}Placement grid re-render failed: "
                            f"{e}{Colors.RESET}"
                        )

                # ------- poll async data-check result -------
                if pending_check_future is not None and pending_check_future.done():
                    try:
                        report = pending_check_future.result()
                        ep_id = pending_check_episode_id
                        episode_key = f"episode_{ep_id}{PENDING_SUFFIX}"
                        findings = data_check_findings(report["episodes"][episode_key])
                        for line in warning_lines(
                            report["episodes"][episode_key].get("warnings", [])
                        ):
                            click.echo(line)

                        if report["episodes"][episode_key]["passed"]:
                            if recorder_manager.finalize_episode(findings=findings):
                                click.echo(
                                    f"{Colors.GREEN}[+] Data check PASSED - Episode {ep_id} finalized{Colors.RESET}"
                                )
                                visualizer.flash_timer_pass()
                            else:
                                click.echo(
                                    f"{Colors.RED}Warning: Failed to finalize episode{Colors.RESET}"
                                )
                                error_state = {
                                    "title": "SAVE ERROR: Failed to finalize episode",
                                    "issues": [
                                        "Episode data could not be written to disk"
                                    ],
                                    "fix_hint": "Check disk space and permissions -- press R to resume",
                                }
                                visualizer.trigger_shake()
                        else:
                            click.echo(
                                f"{Colors.RED}[X] Data check FAILED for episode "
                                f"{ep_id}:{Colors.RESET}"
                            )
                            issues = report["episodes"][episode_key]["issues"]
                            for line in terminal_lines(issues):
                                click.echo(line)

                            if recorder_manager.quarantine_staging(
                                issues=issues, findings=findings
                            ):
                                click.echo(
                                    f"{Colors.YELLOW}[!] Episode {ep_id} quarantined (moved to _quarantine dataset){Colors.RESET}"
                                )
                                error_state = {
                                    "title": "DATA CHECK FAILED -- episode quarantined",
                                    "issues": overlay_lines(issues),
                                    "fix_hint": overlay_hint(issues),
                                }
                                visualizer.trigger_shake()
                            else:
                                click.echo(
                                    f"{Colors.RED}Warning: Failed to quarantine episode. Manual cleanup may be required.{Colors.RESET}"
                                )
                                error_state = {
                                    "title": "QUARANTINE ERROR: Manual cleanup required",
                                    "issues": overlay_lines(issues)
                                    + ["Automatic quarantine failed"],
                                    "fix_hint": "Manually remove episode data -- press R to resume",
                                }
                                visualizer.trigger_shake()
                    except Exception as e:
                        click.echo(f"{Colors.RED}Data check error: {e}{Colors.RESET}")
                        recorder_manager.discard_staging()
                        error_state = {
                            "title": "CHECK ERROR: Episode discarded",
                            "issues": [str(e)],
                            "fix_hint": "Check system state -- press R to resume",
                        }
                        visualizer.trigger_shake()
                    finally:
                        pending_check_future = None
                        pending_check_episode_id = None

                _t0 = time.perf_counter()
                frames = recorder_manager.get_latest_frames()
                if _timing_on:
                    _t_acc["frames"] += time.perf_counter() - _t0

                gripper_lost: list[str] = []
                if recorder_manager.is_recording:
                    now_m = time.monotonic()
                    if now_m - _last_gripper_check_m >= _gripper_check_interval_s:
                        _last_gripper_check_m = now_m
                        try:
                            assert_all_uart_readers_connected()
                        except AssertionError as exc:
                            # A gripper unplug is a lost device, not a bug. Hand
                            # it to the same error_state a lost camera uses,
                            # instead of letting it crash the recorder -- a crash
                            # kills the preview window and leaves the station's
                            # Stop button with a dead process to signal.
                            gripper_lost.append(str(exc))

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
                        elif isinstance(frame, (NumericFrame, TrackingFrame)):
                            numeric_data[name] = frame_list

                # Collect active frame-drop/delay warnings from all recorders.
                # Shake once when a warning first appears; the overlay persists
                # until the condition clears on its own.
                current_warnings = {}
                for recorder in recorders:
                    current_warnings.update(recorder.get_active_warnings())
                new_warning_keys = set(current_warnings.keys()) - prev_warning_keys
                if new_warning_keys:
                    visualizer.trigger_shake(duration=0.8, intensity=12)
                prev_warning_keys = set(current_warnings.keys())

                # Auto-quarantine on device disconnect (source_lost). A gripper
                # that dropped its UART link joins the same list, so a lost
                # gripper and a lost camera share one graceful path.
                lost_sources = [
                    (src, msg)
                    for (src, wtype), msg in current_warnings.items()
                    if wtype == "source_lost"
                ]
                lost_sources += [("gripper", msg) for msg in gripper_lost]
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

                if visualizer.config.enabled and video_frames:
                    _t0 = time.perf_counter()
                    combined = visualizer.combine_frames(
                        video_frames, labels=video_source_order
                    )
                    if _timing_on:
                        _t_acc["combine"] += time.perf_counter() - _t0

                    _t0 = time.perf_counter()
                    # Add overlay
                    combined = visualizer.draw_overlay(
                        combined,
                        episode_id=recorder_manager.episode_id,
                        is_recording=recorder_manager.is_recording,
                        numeric_data=numeric_data if numeric_recorders else None,
                        pose_data=pose_data if pose_data else None,
                        error_active=error_state is not None,
                    )

                    # Pre-episode overlays (e.g. the calibration progress panel)
                    # stay visible in all non-recording stages (IDLE,
                    # CALIBRATION and INIT_POSE), so the operator has live
                    # feedback right up to the recording-start press. The
                    # inline-calibration tick below is still gated on the
                    # CALIBRATION stage, so it never runs during INIT_POSE.
                    if not recorder_manager.is_recording:
                        check_ctx = PreEpisodeContext(
                            latest_frames=frames,
                            cfg=cfg,
                            recorder_manager=recorder_manager,
                            video_source_order=video_source_order,
                        )

                        # Drive inline calibration every frame while active.
                        if (
                            stage == RecordingStage.CALIBRATION
                            and calibration_check is not None
                        ):
                            stage = _tick_calibration_stage(
                                calibration_check, check_ctx, visualizer
                            )

                        for chk in pre_episode_checks:
                            if hasattr(chk, "draw"):
                                chk.draw(combined, check_ctx)

                    # Add task description display
                    combined = visualizer.draw_task_display(
                        combined, recorder_manager.task_description
                    )

                    # Apply any active shake to the base frame before drawing
                    # warning / error overlays. No-op when no shake is active.
                    # Used by error states, frame-drop warnings,
                    # and the calibration "not ready" gate-fail.
                    combined = visualizer.apply_shake(combined)

                    # Non-blocking warning overlay for frame drop/delay issues.
                    # Drawn below the error overlay so errors always take precedence.
                    if current_warnings and error_state is None:
                        combined = visualizer.draw_warning_overlay(
                            combined, current_warnings
                        )

                    if error_state is not None:
                        combined = visualizer.draw_error_overlay(
                            combined,
                            title=error_state["title"],
                            issues=error_state["issues"],
                            fix_hint=error_state["fix_hint"],
                        )

                    if _timing_on:
                        _t_acc["overlay"] += time.perf_counter() - _t0

                    _t0 = time.perf_counter()
                    key = visualizer.show(combined)
                    if _timing_on:
                        _t_acc["show"] += time.perf_counter() - _t0
                else:
                    # No video, just poll for keys
                    key = cv2.waitKey(1) & 0xFF

                # When a blocking error is active, only R (acknowledge) and Q (quit)
                # are processed; all other keys are silently ignored.
                if error_state is not None and key != ord("q"):
                    if key == ord("r") and not error_state.get("fatal", False):
                        error_state = None
                        # Reset any in-flight calibration session so a stale
                        # DONE/progress overlay doesn't linger past the error
                        # ack.
                        if calibration_check is not None:
                            calibration_check.reset_session()
                        stage = RecordingStage.IDLE
                        click.echo(
                            f"{Colors.BLUE}Error acknowledged -- ready to record.{Colors.RESET}"
                        )
                    continue

                if key == ord("q"):
                    click.echo("\nQuitting...")
                    if recorder_manager.is_recording:
                        recorder_manager.stop_recording()
                        recorder_manager.clear_recording()
                    recorder_manager.discard_staging()
                    break

                elif key == ord("g"):
                    if recorder_manager.is_recording:
                        click.echo("Cannot change layout while recording")
                    elif placement_episode_id is None:
                        click.echo("No placement grid to re-roll for this task")
                    else:
                        # Regenerate an unusable placement layout. Seed by session,
                        # episode and
                        # roll number so each result remains reproducible.
                        layout_reroll += 1
                        new_layout = sample_layout(
                            operator_instruction,
                            objects=scene_objects,
                            seed=_layout_seed(
                                layout_session_uid,
                                f"{placement_episode_id}:r{layout_reroll}",
                            ),
                        )
                        if new_layout is not None:
                            recorder_manager.placement_layout = new_layout
                            visualizer.update_filler_cell(
                                1,
                                {
                                    "array": render_layout(new_layout),
                                    "title": "Placement Grid",
                                },
                            )
                            click.echo(
                                f"Placement layout re-rolled (episode "
                                f"{placement_episode_id}, roll {layout_reroll})"
                            )

                elif key == ord("s"):
                    if recorder_manager.is_recording:
                        click.echo("Already recording")
                    elif pending_check_future is not None:
                        click.echo("Data check still running, please wait...")
                    elif stage == RecordingStage.CALIBRATION:
                        if (
                            calibration_check is not None
                            and calibration_check.is_session_converged()
                        ):
                            # Acknowledge calibration completion before moving to
                            # INIT_POSE. Run the
                            # pre-episode gate on the recording-start press, not on this
                            # transition.
                            calibration_check.reset_session()
                            stage = _enter_init_pose()
                        else:
                            click.echo(
                                f"{Colors.YELLOW}[calibration] Calibration in progress — "
                                f"hold the marker visible to all cameras until DONE appears.{Colors.RESET}"
                            )
                    elif stage == RecordingStage.IDLE:
                        stage = _advance_past_idle()
                    elif stage == RecordingStage.INIT_POSE:
                        # Validate device readiness immediately before recording starts.
                        # A failed
                        # gate leaves the operator in INIT_POSE to adjust and retry.
                        check_warnings = []
                        ok, failures = run_pre_episode_checks(
                            pre_episode_checks,
                            PreEpisodeContext(
                                latest_frames=recorder_manager.get_latest_frames(),
                                cfg=cfg,
                                recorder_manager=recorder_manager,
                            ),
                            check_warnings,
                        )
                        for cname, msg in check_warnings:
                            click.echo(
                                f"{Colors.YELLOW}[pre-check] {cname} (warn only): {msg}{Colors.RESET}"
                            )
                        if not ok:
                            for cname, msg in failures:
                                click.echo(
                                    f"{Colors.RED}[pre-check] {cname}: {msg}{Colors.RESET}"
                                )
                        else:
                            visualizer.start_timer()
                            visualizer.reset_fps_counter()
                            if recorder_manager.reset_episode_recording():
                                recorder_manager.start_recording()
                                click.echo(
                                    f"{Colors.BLUE}{Colors.BOLD}Recording started - Episode {recorder_manager.episode_id}{Colors.RESET}"
                                )
                                stage = RecordingStage.IDLE

                elif key == ord("w"):
                    if (
                        stage == RecordingStage.CALIBRATION
                        and calibration_check is not None
                    ):
                        click.echo(
                            f"{Colors.YELLOW}[calibration] Aborting inline calibration.{Colors.RESET}"
                        )
                        calibration_check.reset_session()
                        stage = RecordingStage.IDLE
                    if pending_check_future is not None:
                        click.echo("Data check still running, please wait...")
                    elif recorder_manager.is_recording:
                        visualizer.stop_timer()
                        click.echo("Saving recording to staging...")
                        if recorder_manager.stop_recording():
                            # Save to pending folder (episode_N.pending/)
                            recorder_manager.save_recordings(pending=True)
                            pending_check_episode_id = recorder_manager.episode_id
                            click.echo(
                                f"{Colors.BLUE}{Colors.BOLD}Episode {pending_check_episode_id} saved to staging{Colors.RESET}"
                            )
                            total_hours = recorder_manager.get_total_data_hours()
                            click.echo(
                                f"{Colors.PURPLE}Total data collected: {total_hours:.2f} hours{Colors.RESET}"
                            )

                            if ignore_data_check:
                                click.echo("Ignoring data check")
                                recorder_manager.finalize_episode()
                                click.echo("Episode finalized without data check")
                            else:
                                # Run the data quality check in a subprocess so the
                                # GIL stays free and camera stream threads are not starved.
                                click.echo("Running data quality check (subprocess)...")
                                pending_check_future = check_executor.submit(
                                    check_dataset,
                                    data_dir=data_path,
                                    episodes=[pending_check_episode_id],
                                    check_config_path=os.path.expanduser(check_config),
                                    pending=True,
                                )
                    else:
                        stage = RecordingStage.IDLE
                        click.echo("Not recording -- stage reset to IDLE")

                elif key == ord("a"):
                    if (
                        stage == RecordingStage.CALIBRATION
                        and calibration_check is not None
                    ):
                        click.echo(
                            f"{Colors.YELLOW}[calibration] Aborting inline calibration.{Colors.RESET}"
                        )
                        calibration_check.reset_session()
                    if recorder_manager.is_recording:
                        visualizer.stop_timer()
                        click.echo(
                            f"{Colors.BLUE}{Colors.BOLD}Restarting episode...{Colors.RESET}"
                        )
                        recorder_manager.stop_recording()
                        recorder_manager.clear_recording()
                        visualizer.reset_fps_counter()
                        stage = RecordingStage.IDLE
                    else:
                        stage = RecordingStage.IDLE
                        click.echo(
                            f"{Colors.BLUE}{Colors.BOLD}Not recording -- stage reset to IDLE{Colors.RESET}"
                        )

    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")

    finally:
        # Ignore repeated SIGTERM once shutdown begins so it cannot interrupt the
        # alignment drain. SIGKILL escalation remains available for a stalled drain.
        # Apply this guard even when shutdown began through the q key.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Report the shutdown phase: closing hardware can stall on a lost device,
        # while alignment draining may legitimately be silent for minutes.
        click.echo(shutdown_progress.marker(shutdown_progress.TEARDOWN))
        recorder_manager.stop_recording()
        recorder_manager.stop_streaming()
        # Settle any saved episode's pending data check before discarding staging.
        # Finalize or quarantine it first so discard_staging cannot delete saved data.
        _settle_pending_check(
            recorder_manager, pending_check_future, pending_check_episode_id
        )
        # Discard only an interrupted episode that was never saved.
        recorder_manager.discard_staging()
        visualizer.close()
        stop_all_uart_readers()
        stop_all_oak_devices()
        check_executor.shutdown(wait=False)
        # All hardware is released before the alignment drain; remaining work is disk
        # I/O.
        click.echo(shutdown_progress.marker(shutdown_progress.DRAIN))
        # Drain pending alignments before exit.
        recorder_manager.shutdown_background_services()
        click.echo(shutdown_progress.marker(shutdown_progress.DONE))
        click.echo("Recording stopped")
        # Log unexpected exceptions while the session tee is installed. The later
        # os._exit bypasses Python's default exception reporting.
        exc = sys.exc_info()[1]
        if exc is not None:
            traceback.print_exception(exc)


if __name__ == "__main__":
    main()
