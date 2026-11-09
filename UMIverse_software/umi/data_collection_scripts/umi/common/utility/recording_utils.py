# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Recording utility functions for creating video/numeric recorders from config.

This module provides factory functions and utilities for setting up video and numeric
recorders from configuration dictionaries, reducing code duplication across scripts.
"""

import os
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from umi.common.data.frame import (
    BUILTIN_FRAME_NAMES,
    FRAME_CLASS_MAP,
    serialize_frame_definition,
)
from umi.common.recording import NumericRecorder, VideoRecorder
from umi.common.recording.video_writer_backends import VideoEncoderKind
from umi.common.utility.gripper_pipeline import GripperPipeline
from umi.common.utility.print_utils import print_red, print_yellow

_VALID_VIDEO_ENCODERS: tuple[str, ...] = ("opencv_mp4v", "ffmpeg_h264")

# Preserve placement and sensor_module as descriptive camera provenance.
# Read stream fields with a source fallback to support multi-stream boards
# and single-camera sources. Omit unknown values rather than recording nulls.
_VIDEO_STREAM_META_KEYS: tuple[str, ...] = (
    "placement",
    "sensor_module",
)


class DeviceConstructionError(RuntimeError):
    """A declared source could not be constructed.

    Recording aborts rather than continuing: a source that fails to construct is
    dropped from the expanded config and never recorded, so the saved config and
    the recording agree and no downstream check can tell the session apart from a
    correct one. Failing here is the last moment this is cheap to fix.
    """


def _source_label(source_cfg: Any) -> str:
    """Best-effort operator-facing name for a source that failed to construct.

    ``oak_device`` blocks name their streams rather than the block itself, so
    fall back to the stream names and then to the device identifier.

    Total by construction: this runs *inside* an exception handler, so raising
    here would escape uncaught and defeat ``allow_missing_devices``. A hand-edited
    YAML that leaves a bare ``-`` or a scalar under ``video_sources:`` yields a
    ``None``/``str``/``int`` rather than a mapping.
    """
    if not isinstance(source_cfg, dict):
        return f"<malformed source entry {source_cfg!r}>"
    name = source_cfg.get("name")
    if name:
        return str(name)
    stream_names = [
        str(s.get("name"))
        for s in (source_cfg.get("streams") or [])
        if isinstance(s, dict) and s.get("name")
    ]
    if stream_names:
        return "+".join(stream_names)
    for key in ("oak_device_name", "oak_device_id"):
        if source_cfg.get(key):
            return str(source_cfg[key])
    return f"<unnamed {source_cfg.get('type', 'source')}>"


def _source_failure(source_cfg: Any, cause: Any) -> str:
    """One bullet for the abort summary. Must not raise — see ``_source_label``."""
    source_type = source_cfg.get("type") if isinstance(source_cfg, dict) else None
    return f"{_source_label(source_cfg)} (type={source_type}): {cause}"


def _is_mandatory(source_cfg: Any, require_all_sources: bool) -> bool:
    """Whether this source's failure aborts even under ``allow_missing_devices``.

    Must not raise — see ``_source_label``: this also runs inside the exception
    handler, where a malformed entry is not a dict and has no ``.get``.
    """
    if require_all_sources:
        return True
    return isinstance(source_cfg, dict) and bool(source_cfg.get("required", False))


def _device_construction_message(
    kind: str, failures: List[str], *, mandatory: bool = False
) -> str:
    listed = "\n".join(f"  - {f}" for f in failures)
    # A mandatory-source failure requires fixing the device or config;
    # allow_missing_devices cannot waive it.
    remedy = (
        "Fix the device, or drop it from the config for an intentionally "
        "reduced rig — this config marks its sources mandatory "
        "(require_all_sources / required), so --allow-missing-devices does not "
        "apply."
        if mandatory
        else "Fix the device, or record anyway with --allow-missing-devices — the "
        "session will be missing that device and the data check cannot detect it."
    )
    return (
        f"{len(failures)} declared {kind} source(s) could not be constructed:\n"
        f"{listed}\n"
        "Recording aborted. A source that fails to construct is dropped from the "
        "saved config.yaml as well as from the recording, so the two agree and no "
        f"downstream data check can tell this session from a correct one. {remedy}"
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file. Supports ~ expansion.

    Returns:
        Configuration dictionary loaded from the YAML file.
    """
    config_path = os.path.expanduser(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Video Source Factory
# =============================================================================


def create_video_source(source_config: Dict[str, Any]) -> List:
    """Factory function to create video sources from config.

    Returns a **list** of Camera instances.  For most camera types this is a
    single-element list.  For ``oak_device`` (multi-stream OAK devices like
    OAK-FFC-3P) the list contains one Camera per configured stream.

    Supports the following camera types:
    - oak_camera: OAK camera (DepthAI) — single stream, backward-compatible
    - oak_device: Multi-stream OAK device (OAK-FFC-3P, OAK-D expanded)
    - mock_camera: Mock camera for testing
    - uvc_camera: USB Video Class camera
    - iphone_camera: iPhone camera via AV Foundation

    Args:
        source_config: Configuration dict with keys:
            - type: Camera type string
            - name: Device name (not required for oak_device)
            - params: Optional dict of additional parameters
            For oak_device, also requires:
            - streams: List of stream config dicts
            - oak_device_name or oak_device_id: Device identifier

    Returns:
        List of Camera instances.

    Raises:
        ValueError: If camera type is unknown.
    """
    source_type = source_config["type"]

    if source_type == "oak_device":
        device = _get_or_create_oak_device(source_config)
        device.build_pipeline(source_config["streams"])
        device.start()
        return device.get_streams()

    name = source_config["name"]
    params = source_config.get("params", {})

    if source_type == "oak_camera":
        from umi.common.hardware_sdk.camera.oak_camera import OAKCamera

        # Recorded data is strict about aspect: a camera_resolution that does
        # not match the sensor refuses to start rather than cropping the view.
        # A source that means to crop says so in its params.
        return [
            OAKCamera(device_name=name, **{"allow_cropped_output": False, **params})
        ]

    elif source_type == "mock_camera":
        from umi.common.hardware_sdk.camera.mock_camera import MockCamera

        return [MockCamera(device_name=name, **params)]

    elif source_type == "uvc_camera":
        from umi.common.hardware_sdk.camera.uvc_camera import UVCCamera

        return [UVCCamera(device_name=name, **params)]

    elif source_type == "iphone_camera":
        from umi.common.hardware_sdk.camera.iphone_camera import IphoneCamera

        return [IphoneCamera(device_name=name, **params)]

    else:
        raise ValueError(f"Unknown video source type: {source_type}")


# =============================================================================
# Numeric Source Factory
# =============================================================================

# Module-level cache for UART readers (keyed by port or device name)
_uart_readers_cache: Dict[str, Any] = {}


def get_uart_reader(
    baudrate: int,
    port: str | None = None,
    uart_device_name: str | None = None,
):
    """Get UART reader from cache or create a new one.

    Readers are cached by port or device name, so multiple devices on the same
    port/device share the reader.

    Args:
        baudrate: Baud rate for the serial connection.
        port: Serial port path (e.g., '/dev/ttyACM0'). Used if uart_device_name not provided.
        uart_device_name: Device name from the device registry (e.g., "UMI-006").
                          If provided, looks up device serial and finds port automatically.

    Returns:
        UartSensorReader instance (cached or newly created).

    Raises:
        ValueError: If neither port nor uart_device_name is provided.
    """
    cache_key = uart_device_name if uart_device_name else port
    if cache_key is None:
        raise ValueError("Either 'port' or 'uart_device_name' must be provided")

    if cache_key in _uart_readers_cache:
        return _uart_readers_cache[cache_key]

    from umi.common.hardware_sdk.sensor import UartSensorReader

    reader = UartSensorReader(
        port=port,
        uart_device_name=uart_device_name,
        baudrate=baudrate,
    )
    reader.start()
    _uart_readers_cache[cache_key] = reader
    print(f"Created UART reader: {cache_key} -> {reader.port}")
    return reader


def assert_all_uart_readers_connected() -> None:
    """Fail fast if any cached UMI UART link is gone or the wrong device is on the port.

    Call this before starting an episode and periodically while recording so unplug
    events surface as a loud terminal error.

    Raises:
        AssertionError: Port missing, reader thread died on SerialException, or
            USB serial no longer matches the configured gripper.
    """
    for key, reader in _uart_readers_cache.items():
        label = reader.uart_device_name or key
        if getattr(reader, "_disconnect_reason", None):
            raise AssertionError(reader._disconnect_reason)
        if not getattr(reader, "port", None):
            raise AssertionError(
                f"[UMI gripper DISCONNECTED] No serial port resolved for {label!r}."
            )
        if not os.path.exists(reader.port):
            raise AssertionError(
                f"[UMI gripper DISCONNECTED] Device {label!r} was on {reader.port}, "
                "but that path no longer exists (USB unplugged or gripper off)."
            )
        reader._verify_device_serial()


def stop_all_uart_readers():
    """Stop all cached UART readers."""
    for port, reader in _uart_readers_cache.items():
        reader.stop()
        print(f"Stopped UART reader on {port}")
    _uart_readers_cache.clear()


def _get_uart_reader_from_config(source_config: Dict[str, Any], device_type: str):
    """Helper to create/get UART reader from source config.

    Args:
        source_config: Source configuration dict with uart_device_name/reader_port/reader_baudrate.
        device_type: Device type string for error messages.

    Returns:
        UartSensorReader instance.

    Raises:
        ValueError: If required fields are missing.
    """
    name = source_config["name"]
    uart_device_name = source_config.get("uart_device_name")
    reader_port = source_config.get("reader_port")
    if uart_device_name is None and reader_port is None:
        raise ValueError(
            f"{device_type} '{name}' requires either 'uart_device_name' or 'reader_port' field"
        )
    reader_baudrate = source_config.get("reader_baudrate")
    if reader_baudrate is None:
        raise ValueError(f"{device_type} '{name}' requires 'reader_baudrate' field")
    return get_uart_reader(
        port=reader_port,
        uart_device_name=uart_device_name,
        baudrate=reader_baudrate,
    )


# =============================================================================
# OAK Device Cache (multi-stream devices)
# =============================================================================

# Module-level cache for OAK devices (keyed by device name or ID)
_oak_device_cache: Dict[str, Any] = {}


def _get_or_create_oak_device(source_config: Dict[str, Any]):
    """Get an OAK device from cache or create a new one.

    Args:
        source_config: Source configuration dict with oak_device_name/oak_device_id.

    Returns:
        OAKDevice instance.
    """
    from umi.common.hardware_sdk.camera.oak_device import OAKDevice

    device_name = source_config.get("oak_device_name")
    device_id = source_config.get("oak_device_id")
    cache_key = device_id or device_name
    if cache_key and cache_key in _oak_device_cache:
        return _oak_device_cache[cache_key]

    device = OAKDevice(
        oak_device_name=device_name,
        oak_device_id=device_id,
        fps=source_config.get("fps", 30),
        latency=source_config.get("latency", 0.0),
        ae_max_exposure_us=source_config.get("ae_max_exposure_us"),
        # Require explicit opt-in for cropped recording output.
        allow_cropped_output=bool(source_config.get("allow_cropped_output", False)),
    )
    if cache_key:
        _oak_device_cache[cache_key] = device
    return device


def stop_all_oak_devices():
    """Stop all cached OAK devices."""
    for key, device in _oak_device_cache.items():
        device.stop()
        print(f"Stopped OAK device: {key}")
    _oak_device_cache.clear()


def release_constructed_devices(sources: Iterable[Any] = ()) -> None:
    """Close every device opened so far, before a construction abort escapes.

    The entry points' ``try/finally`` teardown only wraps the *recording loop* —
    it starts after recorder construction — so nothing else runs on this path.
    Without this, aborting on the second camera exits with the first one's
    ``dai.Device`` still open; the operator fixes the dead camera, retries
    immediately, and the retry fails on a *different* camera that has not
    finished re-enumerating. One dead camera becomes two, exactly when someone is
    already retrying under pressure.

    Two mechanisms, because two kinds of device exist:

    - *sources* — the instances constructed so far in the aborting call. A
      ``type: oak_camera`` block opens its ``dai.Device`` inside
      ``OAKCamera.__init__`` and the instance is held nowhere but the caller's
      local list, so draining the caches alone would miss it. An optional head
      or bottom OAK (``production/optional_devices.yaml``) is one.
    - the module caches — ``oak_device`` blocks share one ``OAKDevice`` via
      ``_oak_device_cache`` and grippers share UART readers via
      ``_uart_readers_cache``; their per-stream ``stop_streaming`` is only a flag
      flip, so the cache drain is what actually closes them.

    Best-effort throughout: a teardown error must never mask the
    DeviceConstructionError that names the device the operator has to fix.
    """
    for source in sources:
        stop = getattr(source, "stop_streaming", None)
        if stop is None:
            continue
        try:
            stop()
        except Exception as e:
            name = getattr(source, "device_name", source)
            print_red(f"Warning: failed to release {name} during abort: {e}")

    for release in (stop_all_oak_devices, stop_all_uart_readers):
        try:
            release()
        except Exception as e:
            print_red(f"Warning: failed to release devices during abort: {e}")


# =============================================================================
# Numeric Source Factory
# =============================================================================


def create_numeric_source(source_config: Dict[str, Any]) -> List:
    """Factory function to create numeric data sources from config.

    Creates the device and calls get_data_sources() to return individual
    SensorInterface data sources for the recorder.

    Supports the following device types:
    - umi_gripper: UMI gripper with UART reader
    - vive_tracker: Vive tracker pose over the vive server
    - mock_sensor: Mock sensor for testing (returned directly as a data source)

    Args:
        source_config: Configuration dict with keys:
            - type: Device type string
            - name: Device name
            - params: Optional dict of additional parameters
            - uart_device_name: Preferred, device name from the device registry
            - reader_port: Legacy option, serial port path
            - reader_baudrate: Required for UART devices, baud rate

    Returns:
        List of SensorInterface data sources.

    Raises:
        ValueError: If source type is unknown or required fields are missing.
    """
    source_type = source_config["type"]
    name = source_config["name"]
    params = source_config.get("params", {})

    if source_type == "umi_gripper":
        from umi.umi_device.umi_device_interface import create_umi_device
        from umi.umi_device.umi_gripper import UmiGripper

        reader = _get_uart_reader_from_config(source_config, "umi_gripper")
        device = create_umi_device(
            device_class=UmiGripper,
            device_name=name,
            reader=reader,
            **params,
        )
        return device.get_data_sources()

    elif source_type == "vive_tracker":
        from umi.common.hardware_sdk.vive import ViveTracker

        # Require poll_hz above stream_fps: repeated held timestamps do not count
        # as fresh frames and can exhaust the recorder's timeout budget.
        poll_hz = float(params.get("poll_hz", 90.0))
        stream_fps = source_config.get("stream_fps")
        if stream_fps is not None and poll_hz <= float(stream_fps):
            raise ValueError(
                f"vive_tracker {name!r}: poll_hz ({poll_hz:g}) must be greater "
                f"than stream_fps ({float(stream_fps):g}), or the recorder "
                "starves on samples it has already consumed."
            )
        if "tracking" not in name.lower():
            # Warn when configured pose sources are excluded from tracking checks;
            # recording may be intentional, but a green report must not imply coverage.
            print_yellow(
                f"vive_tracker {name!r}: name has no 'tracking' substring, so "
                "the data check will NOT run pose_movement / pose_jumps / "
                "consecutive_movement on it (dispatch.py selects by that "
                "substring). Rename to e.g. 'left_tracking_pose' to enable them."
            )
        # Pass consumer rate to the client so it can warn when the server cannot
        # supply fresh samples fast enough, rather than when achieved rate falls
        # below the deliberately high poll target.
        return [
            ViveTracker(
                device_name=name,
                expected_read_hz=float(stream_fps) if stream_fps else 0.0,
                **params,
            )
        ]

    elif source_type == "mock_sensor":
        from umi.common.hardware_sdk.sensor.mock_sensor import MockSensor

        return [MockSensor(device_name=name, **params)]

    else:
        raise ValueError(
            f"Unknown numeric source type: {source_type}. "
            "Add support in create_numeric_source() function."
        )


# =============================================================================
# Video Recorder Creation
# =============================================================================


def create_video_recorders(
    video_source_configs: List[Dict[str, Any]],
    record_path: str,
    default_record_fps: int = 30,
    default_stream_fps: int = 60,
    default_frame_type: str = "RGBCameraFrame",
    verbose: bool = True,
    info_verbose: bool = True,
    warning_verbose: bool = True,
    video_encoder: VideoEncoderKind | str = "ffmpeg_h264",
    allow_missing_devices: bool = False,
    require_all_sources: bool = False,
) -> Tuple[List[VideoRecorder], List, List[Dict[str, Any]]]:
    """Create VideoRecorder instances from config, grouping by fps/frame_type.

    Sources with the same (record_fps, stream_fps, frame_type) settings are
    grouped into a single VideoRecorder for efficiency.

    Args:
        video_source_configs: List of video source configuration dicts.
        record_path: Path to save recordings.
        default_record_fps: Default recording FPS if not specified per-source.
        default_stream_fps: Default streaming FPS if not specified per-source.
        default_frame_type: Default frame type class name.
        verbose: Enable verbose logging.
        video_encoder: "ffmpeg_h264" (smaller files, default) or "opencv_mp4v" (fast).
        allow_missing_devices: Build recorders even if a declared camera cannot be
            opened — the session will be missing that camera and the data check
            cannot detect it. Default False: every failure is collected and
            raised together as ``DeviceConstructionError``.
        require_all_sources: Mark every source in this config as mandatory, so
            that ``allow_missing_devices`` does not excuse it. A per-source
            ``required: true`` key does the same for one source. Both are no-ops
            under the default (which already aborts on any failure); they exist
            so a rig that is meaningless when partial — a bimanual session
            missing one side, say — cannot be recorded even by an operator who
            reached for the escape hatch.

    Returns:
        Tuple of (list of VideoRecorder instances, list of created sources,
        list of expanded source config dicts).

    Raises:
        DeviceConstructionError: If any declared source failed to construct and
            ``allow_missing_devices`` is False, or if a mandatory source failed
            regardless of ``allow_missing_devices``.
    """
    if video_encoder not in _VALID_VIDEO_ENCODERS:
        raise ValueError(
            f"video_encoder must be one of {_VALID_VIDEO_ENCODERS}, got {video_encoder!r}"
        )
    video_source_groups: Dict[tuple, List] = {}
    all_sources = []
    expanded_configs: List[Dict[str, Any]] = []
    # Collect initialization failures so one failed launch reports every
    # missing or unusable source.
    failures: List[str] = []
    # Set when a source the config declares mandatory failed — that aborts even
    # under allow_missing_devices.
    mandatory_failed = False

    for source_cfg in video_source_configs:
        try:
            sources = create_video_source(source_cfg)
            # Treat an empty factory result as initialization failure; silent source
            # loss must not yield an apparently complete recording.
            if not sources:
                raise RuntimeError("source factory returned no video streams")

            # For oak_device, per-stream settings come from each stream config;
            # for all other types, sources is a single-element list and the
            # top-level source_cfg holds all settings.
            if source_cfg["type"] == "oak_device":
                stream_cfgs = source_cfg["streams"]
            else:
                stream_cfgs = [source_cfg] * len(sources)

            for source, stream_cfg in zip(sources, stream_cfgs):
                all_sources.append(source)

                record_fps = stream_cfg.get(
                    "record_fps",
                    source_cfg.get("record_fps", default_record_fps),
                )
                stream_fps = stream_cfg.get(
                    "stream_fps",
                    source_cfg.get("stream_fps", default_stream_fps),
                )
                frame_type_name = stream_cfg.get(
                    "frame_type",
                    source_cfg.get("frame_type", default_frame_type),
                )

                if frame_type_name not in FRAME_CLASS_MAP:
                    raise ValueError(f"Unknown frame type: {frame_type_name}")

                group_key = (record_fps, stream_fps, frame_type_name)
                if group_key not in video_source_groups:
                    video_source_groups[group_key] = []
                video_source_groups[group_key].append(source)
                print(f"  Video source: {source.device_name}")

                entry = {
                    "type": source_cfg["type"],
                    "name": source.device_name,
                    "frame_type": frame_type_name,
                    "latency": stream_cfg.get(
                        "latency", source_cfg.get("latency", 0.0)
                    ),
                    "record_fps": record_fps,
                    "stream_fps": stream_fps,
                }
                if "required" in source_cfg:
                    entry["required"] = bool(source_cfg["required"])
                # Preserve camera_resolution when expanding multi-stream OAK sources.
                params = stream_cfg.get("params")
                if params:
                    entry["params"] = params
                elif source_cfg["type"] == "oak_device":
                    resolution = stream_cfg.get("resolution")
                    if resolution:
                        entry["params"] = {"camera_resolution": resolution}
                # Carry descriptive camera metadata through the shared stream/source
                # lookup.
                for _meta_key in _VIDEO_STREAM_META_KEYS:
                    _meta_value = stream_cfg.get(_meta_key, source_cfg.get(_meta_key))
                    if _meta_value:
                        entry[_meta_key] = _meta_value
                expanded_configs.append(entry)

        except Exception as e:
            label = _source_label(source_cfg)
            # Retain the traceback with the summary so the device-open cause is visible.
            print_red(f"Error: Failed to create video source {label}: {e}")
            print(traceback.format_exc())
            failures.append(_source_failure(source_cfg, e))
            mandatory_failed = mandatory_failed or _is_mandatory(
                source_cfg, require_all_sources
            )

    if failures and (not allow_missing_devices or mandatory_failed):
        # Track every opened camera for cleanup, including devices outside shared
        # caches.
        release_constructed_devices(all_sources)
        raise DeviceConstructionError(
            _device_construction_message("video", failures, mandatory=mandatory_failed)
        )

    recorders = []
    for (
        record_fps,
        stream_fps,
        frame_type_name,
    ), sources in video_source_groups.items():
        frame_class = FRAME_CLASS_MAP[frame_type_name]
        recorder = VideoRecorder(
            record_fps=record_fps,
            stream_fps=stream_fps,
            record_path=record_path,
            data_sources=sources,
            frame_data_class=frame_class,
            verbose=verbose,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
            video_encoder=video_encoder,
        )
        recorders.append(recorder)

    return recorders, all_sources, expanded_configs


# =============================================================================
# Numeric Recorder Creation
# =============================================================================


def create_numeric_recorders(
    numeric_source_configs: List[Dict[str, Any]],
    record_path: str,
    default_record_fps: int = 30,
    default_stream_fps: int = 60,
    default_frame_type: str = "NumericFrame",
    verbose: bool = False,
    info_verbose: bool = True,
    warning_verbose: bool = True,
    recording_meta: Optional[Dict[str, Any]] = None,
    allow_missing_devices: bool = False,
    require_all_sources: bool = False,
) -> Tuple[List[NumericRecorder], List, List[Dict[str, Any]]]:
    """Create NumericRecorder instances from config, grouping by fps/frame_type.

    Sources with the same (record_fps, stream_fps, frame_type) settings are
    grouped into a single NumericRecorder for efficiency.

    Args:
        numeric_source_configs: List of numeric source configuration dicts.
        record_path: Path to save recordings.
        default_record_fps: Default recording FPS if not specified per-source.
        default_stream_fps: Default streaming FPS if not specified per-source.
        default_frame_type: Default frame type class name.
        verbose: Enable verbose logging.

    Args:
        recording_meta: Optional ``meta`` block from the recording YAML for visibility only
            (e.g. ``gripper_pipeline`` marker). Does not change numeric source construction.
        allow_missing_devices: Build recorders even if a declared gripper/sensor
            cannot be opened — the session will be missing that device and the
            data check cannot detect it. Default False: every failure is
            collected and raised together as ``DeviceConstructionError``.
        require_all_sources: Mark every source in this config as mandatory, so
            that ``allow_missing_devices`` does not excuse it. See
            ``create_video_recorders`` for the rationale.

    Returns:
        Tuple of (list of NumericRecorder instances, list of all data sources,
        list of expanded source config dicts).

    Raises:
        DeviceConstructionError: If any declared source failed to construct and
            ``allow_missing_devices`` is False, or if a mandatory source failed
            regardless of ``allow_missing_devices``.
    """
    if recording_meta is not None and info_verbose:
        raw_gp = recording_meta.get("gripper_pipeline")
        gp = GripperPipeline.parse(raw_gp)
        if raw_gp is not None and not GripperPipeline.is_known(raw_gp):
            print_red(
                f"Warning: meta.gripper_pipeline={raw_gp!r} is not one of "
                f"{GripperPipeline.values()} — displaying as {gp} for visibility."
            )
        print(
            f"  meta.gripper_pipeline: {gp} "
            "(informational; umi_gripper runtime unchanged)"
        )

    numeric_source_groups: Dict[tuple, List] = {}
    all_sources = []
    expanded_configs: List[Dict[str, Any]] = []
    # See create_video_recorders: collect every failure, report once.
    failures: List[str] = []
    mandatory_failed = False

    for source_cfg in numeric_source_configs:
        try:
            data_sources = create_numeric_source(source_cfg)
            # Treat empty factory results as failed sources, just like constructor
            # errors.
            if not data_sources:
                raise RuntimeError("source factory returned no numeric streams")

            record_fps = source_cfg.get("record_fps", default_record_fps)
            stream_fps = source_cfg.get("stream_fps", default_stream_fps)

            for source in data_sources:
                source_frame_class = getattr(source, "frame_class", None)
                if source_frame_class is not None:
                    frame_type_name = source_frame_class.__name__
                    if frame_type_name not in FRAME_CLASS_MAP:
                        FRAME_CLASS_MAP[frame_type_name] = source_frame_class
                else:
                    frame_type_name = source_cfg.get("frame_type", default_frame_type)

                if frame_type_name not in FRAME_CLASS_MAP:
                    # Exclude failed sources from the expanded config as well as the
                    # recorder list.
                    print_red(
                        f"Error: Unknown frame type {frame_type_name!r} for "
                        f"numeric source {source.device_name}"
                    )
                    # Use the expanded source's name for precise errors within a
                    # multi-source block.
                    failures.append(
                        f"{source.device_name} "
                        f"(type={source_cfg.get('type') if isinstance(source_cfg, dict) else None}): "
                        f"unknown frame type {frame_type_name!r}"
                    )
                    mandatory_failed = mandatory_failed or _is_mandatory(
                        source_cfg, require_all_sources
                    )
                    continue

                group_key = (record_fps, stream_fps, frame_type_name)
                if group_key not in numeric_source_groups:
                    numeric_source_groups[group_key] = []
                numeric_source_groups[group_key].append(source)
                all_sources.append(source)
                print(f"  Numeric source: {source.device_name}")

                entry = {
                    "type": source_cfg.get("type"),
                    "name": source.device_name,
                    "frame_type": frame_type_name,
                    "latency": source.latency
                    if source.latency is not None
                    else source_cfg.get("latency", 0.0),
                    "record_fps": record_fps,
                    "stream_fps": stream_fps,
                }
                # Connection fields belong to the parent device. Preserve
                # them on every expanded sub-source so saved config.yaml
                # retains exact UART provenance.
                for key in (
                    "uart_device_name",
                    "reader_port",
                    "reader_baudrate",
                    "required",
                ):
                    if key in source_cfg:
                        entry[key] = source_cfg[key]
                params = source_cfg.get("params")
                if params:
                    entry["params"] = params
                # Preserve placement metadata on every expanded numeric sub-source.
                placement = source_cfg.get("placement")
                if placement:
                    entry["placement"] = placement
                expanded_configs.append(entry)

        except Exception as e:
            label = _source_label(source_cfg)
            print_red(f"Error: Failed to create numeric source {label}: {e}")
            print(traceback.format_exc())
            failures.append(_source_failure(source_cfg, e))
            mandatory_failed = mandatory_failed or _is_mandatory(
                source_cfg, require_all_sources
            )

    if failures and (not allow_missing_devices or mandatory_failed):
        # Release only sources created in this call. The caller owns cleanup of
        # cameras opened by the earlier video-source pass.
        release_constructed_devices(all_sources)
        raise DeviceConstructionError(
            _device_construction_message(
                "numeric", failures, mandatory=mandatory_failed
            )
        )

    recorders = []
    for (
        record_fps,
        stream_fps,
        frame_type_name,
    ), sources in numeric_source_groups.items():
        frame_class = FRAME_CLASS_MAP[frame_type_name]
        recorder = NumericRecorder(
            record_fps=record_fps,
            stream_fps=stream_fps,
            record_path=record_path,
            numeric_sources=sources,
            frame_data_class=frame_class,
            verbose=verbose,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
        )
        recorders.append(recorder)

    return recorders, all_sources, expanded_configs


def build_frame_definitions(
    expanded_configs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build serialized frame definitions for dynamic (non-built-in) frame classes.

    Args:
        expanded_configs: List of expanded source config dicts, each with
            a "frame_type" key.

    Returns:
        Dict mapping frame class names to their serialized definitions.
    """
    frame_defs: Dict[str, Dict[str, Any]] = {}
    for cfg in expanded_configs:
        frame_type_name = cfg.get("frame_type", "")
        if frame_type_name in BUILTIN_FRAME_NAMES or frame_type_name in frame_defs:
            continue
        frame_class = FRAME_CLASS_MAP.get(frame_type_name)
        if frame_class is not None:
            frame_defs[frame_type_name] = serialize_frame_definition(frame_class)
    return frame_defs
