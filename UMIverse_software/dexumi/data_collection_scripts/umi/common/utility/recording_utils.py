# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Recording utility functions for creating video/numeric recorders from config.

This module provides factory functions and utilities for setting up video and numeric
recorders from configuration dictionaries, reducing code duplication across scripts.
"""

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import yaml

from umi.common.data.frame import (
    BUILTIN_FRAME_NAMES,
    FRAME_CLASS_MAP,
    serialize_frame_definition,
)
from umi.common.recording import NumericRecorder, VideoRecorder
from umi.common.recording.video_writer_backends import VideoEncoderKind
from umi.common.utility.print_utils import print_red

_VALID_VIDEO_ENCODERS: tuple[str, ...] = ("opencv_mp4v", "ffmpeg_h264")


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


def _build_with_identity(source_config: Dict[str, Any], build) -> List:
    """Build one source entry's streams and tag them for identity bookkeeping.

    Every stream learns the config entry it came from (``config_source_name``,
    the key of the DC GUI's device scan). ``mock_*`` entries get a synthetic
    identity; their ``mock_device_id`` / ``mock_observed_device_id`` /
    ``mock_swap_after_reads`` params are stripped before construction.
    """
    from umi.common.recording.device_identity import (
        attach_mock_identity,
        split_mock_identity_params,
    )

    is_mock = str(source_config.get("type") or "").startswith("mock_")
    build_config, knobs = (
        split_mock_identity_params(source_config) if is_mock else (source_config, {})
    )
    sources = build(build_config)
    name = source_config.get("name")
    if name:
        for source in sources:
            try:
                source.config_source_name = name
            except AttributeError:
                pass
    if is_mock and name:
        attach_mock_identity(sources, name, knobs)
    return sources


def create_video_source(source_config: Dict[str, Any]) -> List:
    """Factory function to create video sources from config (see
    ``_create_video_source``), tagged for device-identity bookkeeping."""
    return _build_with_identity(source_config, _create_video_source)


def _create_video_source(source_config: Dict[str, Any]) -> List:
    """Factory function to create video sources from config.

    Returns a single-element **list** of Camera instances.

    Supports the following camera types:
    - oak_camera: OAK camera (DepthAI)
    - mock_camera: Mock camera for testing
    - iphone_camera: iPhone camera via AV Foundation

    Args:
        source_config: Configuration dict with keys:
            - type: Camera type string
            - name: Device name
            - params: Optional dict of additional parameters

    Returns:
        List of Camera instances.

    Raises:
        ValueError: If camera type is unknown.
    """
    source_type = source_config["type"]
    name = source_config["name"]
    params = source_config.get("params", {})

    if source_type == "oak_camera":
        from umi.common.hardware_sdk.camera.oak_camera import OAKCamera

        return [OAKCamera(device_name=name, **params)]

    elif source_type == "mock_camera":
        from umi.common.hardware_sdk.camera.mock_camera import MockCamera

        return [MockCamera(device_name=name, **params)]

    elif source_type == "iphone_camera":
        from umi.common.hardware_sdk.camera.iphone_camera import IphoneCamera

        return [IphoneCamera(device_name=name, **params)]

    else:
        raise ValueError(f"Unknown video source type: {source_type}")


# =============================================================================
# UART Reader Cache
# =============================================================================

# Module-level cache for UART readers (keyed by port or device name)
_uart_readers_cache: Dict[str, Any] = {}

# Numeric source types backed by a UART reader (one shared construction path).
_UART_DEVICE_TYPES = ("dexumi_exoskeleton",)


def _validate_config_device_serial(source_type, name, uart_device_name, config_file):
    """Hard-fail launch when the calibration YAML belongs to a different board.

    calibrate_exo.py records the board's USB serial in the config as
    ``device_serial``; a ``config_file`` other than ``uart_device_name``'s own
    calibration must record the same board. Silently skipped when the field is
    absent or when there is no device name / config file (mock sources,
    raw-port bench use).
    """
    if not uart_device_name or not config_file:
        return
    with open(os.path.expanduser(str(config_file))) as f:
        device_config = yaml.safe_load(f) or {}
    config_serial = device_config.get("device_serial")
    if not config_serial:
        return
    from umi.utils.device_manager import get_exoskeleton_serial

    expected = get_exoskeleton_serial(uart_device_name)
    if str(config_serial).strip() != expected:
        raise ValueError(
            f"{source_type} {name!r}: calibration {str(config_file)!r} records "
            f"device_serial {config_serial}, but device {uart_device_name!r} is "
            f"calibrated on board {expected}. The calibration belongs to a "
            "different physical board — fix uart_device_name or config_file."
        )


def _uart_device_class(source_type: str):
    """Device class for a UART-backed source type (imports kept lazy)."""
    from umi.umi_device.dexumi_exoskeleton import DexUmiExoskeleton

    return DexUmiExoskeleton


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
        uart_device_name: Exoskeleton name (e.g., "DEXUMI-Right"). If
                          provided, the board serial its calibration records
                          selects the port automatically.

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


def get_cached_uart_reader(uart_device_name: str):
    """Return the cached UartSensorReader for this device name, or None."""
    return _uart_readers_cache.get(uart_device_name)


def assert_all_uart_readers_connected() -> None:
    """Fail fast if any cached UMI UART link is gone or the wrong device is on the port.

    Call this before starting an episode and periodically while recording so unplug
    events surface as a loud terminal error.

    Raises:
        AssertionError: Port missing, reader thread died on SerialException, or
            USB serial no longer matches the configured board.
    """
    for key, reader in _uart_readers_cache.items():
        label = reader.uart_device_name or key
        if getattr(reader, "_disconnect_reason", None):
            raise AssertionError(reader._disconnect_reason)
        if not getattr(reader, "port", None):
            raise AssertionError(
                f"[UART DISCONNECTED] No serial port resolved for {label!r}."
            )
        if not os.path.exists(reader.port):
            raise AssertionError(
                f"[UART DISCONNECTED] Device {label!r} was on {reader.port}, "
                "but that path no longer exists (USB unplugged or board off)."
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
# Numeric Source Factory
# =============================================================================


def create_numeric_source(source_config: Dict[str, Any]) -> List:
    """Factory function to create numeric data sources from config (see
    ``_create_numeric_source``), tagged for device-identity bookkeeping."""
    return _build_with_identity(source_config, _create_numeric_source)


def _create_numeric_source(source_config: Dict[str, Any]) -> List:
    """Factory function to create numeric data sources from config.

    Creates the device and calls get_data_sources() to return individual
    SensorInterface data sources for the recorder.

    Supports the following device types:
    - dexumi_exoskeleton: DexUMI exoskeleton with encoder + tactile sensors
    - mock_dexumi_exoskeleton: hardware-free twin of dexumi_exoskeleton

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

    if source_type in _UART_DEVICE_TYPES:
        from umi.umi_device.umi_device_interface import create_umi_device

        device_class = _uart_device_class(source_type)

        # Calibration checks run before creating the UART reader so a failure
        # has no serial side effects.
        given_config = params.get("config_file")
        if given_config:
            device_class.validate_config_file(os.path.expanduser(str(given_config)))

        uart_device_name = source_config.get("uart_device_name")
        canonical = (
            device_class.resolve_config_file(uart_device_name)
            if uart_device_name
            else None
        )
        if canonical:
            if given_config:
                if os.path.basename(str(given_config)) != os.path.basename(canonical):
                    raise ValueError(
                        f"{source_type} {name!r}: config_file {str(given_config)!r} "
                        f"does not belong to device {uart_device_name!r}; expected "
                        f"calibration {canonical!r}. Fix uart_device_name or "
                        "config_file in the recording config."
                    )
            else:
                if not os.path.isfile(canonical):
                    raise FileNotFoundError(
                        f"{source_type} {name!r}: no calibration for device "
                        f"{uart_device_name!r} at {canonical}"
                    )
                # Inject the derived calibration so device creation and the
                # saved session config both carry it.
                params["config_file"] = canonical
                source_config["params"] = params
                device_class.validate_config_file(canonical)

        if source_type == "dexumi_exoskeleton":
            _validate_config_device_serial(
                source_type, name, uart_device_name, params.get("config_file")
            )

        reader = _get_uart_reader_from_config(source_config, source_type)
        # Identity kind of the board behind this reader (device_binding.yaml).
        reader.device_kind = source_type
        device = create_umi_device(
            device_class=device_class,
            device_name=name,
            reader=reader,
            **params,
        )
        return device.get_data_sources()

    elif source_type == "mock_dexumi_exoskeleton":
        # No-hardware twin of dexumi_exoskeleton: same config_file + frame
        # classes (encoder + tactile groups), synthetic data, no UART reader.
        from umi.umi_device.mock_dexumi_exoskeleton import MockDexUmiExoskeleton
        from umi.umi_device.umi_device_interface import create_umi_device

        device = create_umi_device(
            device_class=MockDexUmiExoskeleton,
            device_name=name,
            reader=None,
            **params,
        )
        return device.get_data_sources()

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
    require_all_sources: bool = False,
    frame_queue_size: Optional[int] = None,
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
        require_all_sources: Raise if any configured source cannot be created.
            False preserves the legacy best-effort behavior.
        frame_queue_size: Frames retained per camera in the streaming ring
            buffer (``FrameBuffer`` maxlen). ``None`` (default) keeps the
            ``VideoRecorder`` default. Inference passes the env's
            ``control.frame_buffer_size`` so history observations can reach
            back as far as they fetch.

    Returns:
        Tuple of (list of VideoRecorder instances, list of created sources,
        list of expanded source config dicts).
    """
    if video_encoder not in _VALID_VIDEO_ENCODERS:
        raise ValueError(
            f"video_encoder must be one of {_VALID_VIDEO_ENCODERS}, got {video_encoder!r}"
        )
    video_source_groups: Dict[tuple, List] = {}
    all_sources = []
    expanded_configs: List[Dict[str, Any]] = []

    for source_cfg in video_source_configs:
        try:
            sources = create_video_source(source_cfg)
            if not sources and (
                require_all_sources or source_cfg.get("required", False)
            ):
                raise RuntimeError("source factory returned no video streams")

            for source in sources:
                all_sources.append(source)

                record_fps = source_cfg.get("record_fps", default_record_fps)
                stream_fps = source_cfg.get("stream_fps", default_stream_fps)
                frame_type_name = source_cfg.get("frame_type", default_frame_type)

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
                    "latency": source_cfg.get("latency", 0.0),
                    "record_fps": record_fps,
                    "stream_fps": stream_fps,
                }
                if "required" in source_cfg:
                    entry["required"] = bool(source_cfg["required"])
                params = source_cfg.get("params")
                if params:
                    entry["params"] = params
                expanded_configs.append(entry)

        except Exception as e:
            print_red(f"Error: Failed to create video source: {e}")
            print(traceback.format_exc())
            if require_all_sources or source_cfg.get("required", False):
                source_name = source_cfg.get("name", source_cfg.get("type", "unknown"))
                raise RuntimeError(
                    f"Required video source {source_name!r} could not be created"
                ) from e

    recorders = []
    retention_kwargs = (
        {} if frame_queue_size is None else {"frame_queue_size": frame_queue_size}
    )
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
            **retention_kwargs,
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
        require_all_sources: Raise if any configured source cannot be created.
            False preserves the legacy best-effort behavior.

    Returns:
        Tuple of (list of NumericRecorder instances, list of all data sources,
        list of expanded source config dicts).
    """
    numeric_source_groups: Dict[tuple, List] = {}
    all_sources = []
    expanded_configs: List[Dict[str, Any]] = []

    for source_cfg in numeric_source_configs:
        try:
            data_sources = create_numeric_source(source_cfg)
            if not data_sources and (
                require_all_sources or source_cfg.get("required", False)
            ):
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
                    print_red(f"Unknown frame type: {frame_type_name}")
                    if require_all_sources or source_cfg.get("required", False):
                        raise ValueError(f"Unknown frame type: {frame_type_name}")
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
                # them on every expanded encoder/tactile source so saved
                # config.yaml retains exact UART provenance.
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
                expanded_configs.append(entry)

        except Exception as e:
            print_red(f"Error: Failed to create numeric source: {e}")
            if require_all_sources or source_cfg.get("required", False):
                source_name = source_cfg.get("name", source_cfg.get("type", "unknown"))
                raise RuntimeError(
                    f"Required numeric source {source_name!r} could not be created"
                ) from e

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
