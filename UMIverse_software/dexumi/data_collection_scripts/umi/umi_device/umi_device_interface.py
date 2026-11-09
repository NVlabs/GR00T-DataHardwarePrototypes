# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

import numpy as np
import yaml

from umi.common.data.frame import Frame
from umi.common.hardware_sdk.sensor_interface import SensorInterface

T = TypeVar("T", bound="UmiDeviceInterface")


def create_data_source(
    sensor: SensorInterface,
    device_name: str | None = None,
    latency: float | None = None,
    frame_class: Type[Frame] | None = None,
    get_frame_fn: Callable[[], Frame | None] | None = None,
) -> SensorInterface:
    """Create a data source that wraps an existing sensor.

    Pure delegation by default. Pass get_frame_fn to override get_frame()
    with custom logic (e.g. angle offset computation, unit conversion).

    Args:
        sensor: The underlying SensorInterface to delegate start/stop to.
        device_name: Override device name (default: sensor's device_name).
        latency: Override latency (default: sensor's latency).
        frame_class: Override frame class (default: sensor's frame_class).
        get_frame_fn: Custom get_frame callable. If None, delegates to sensor.

    Returns:
        A SensorInterface instance wrapping the sensor.
    """
    return DynamicSourceWrapper(
        sensor=sensor,
        device_name=device_name,
        latency=latency,
        frame_class=frame_class,
        get_frame_fn=get_frame_fn,
    )


class DynamicSourceWrapper(SensorInterface):
    """Generic data source that wraps a sensor with optional get_frame override."""

    def __init__(
        self,
        sensor: SensorInterface,
        device_name: str | None = None,
        latency: float | None = None,
        frame_class: Type[Frame] | None = None,
        get_frame_fn: Callable[[], Frame | None] | None = None,
    ) -> None:
        self._sensor = sensor
        self._get_frame_fn = get_frame_fn
        self.frame_class = frame_class
        super().__init__(
            device_name=device_name or sensor.device_name,
            latency=latency if latency is not None else sensor.latency,
        )

    def start_streaming(self) -> None:
        self._sensor.start_streaming()

    def stop_streaming(self) -> None:
        self._sensor.stop_streaming()

    def get_frame(self) -> Frame | None:
        if self._get_frame_fn is not None:
            return self._get_frame_fn()
        return self._sensor.get_frame()

    def identity(self) -> dict | None:
        getter = getattr(self._sensor, "identity", None)
        return getter() if callable(getter) else None

    def observed_identity(self) -> dict | None:
        getter = getattr(self._sensor, "observed_identity", None)
        return getter() if callable(getter) else None


def create_umi_device(
    device_class: type[T],
    device_name: str,
    reader: Any,
    config_file: str | Path | None = None,
    **params: Any,
) -> T:
    """Generic factory for UMI devices.

    Handles the common pattern:
    1. Load from config file if exists, otherwise create with params

    Note: Calibration is handled by each device's __init__ method if needed.

    Args:
        device_class: The UmiDeviceInterface subclass to instantiate
        device_name: Unique device identifier
        reader: Shared UartSensorReader instance
        config_file: Optional path to config file (loads if exists)
        **params: Device-specific parameters (used if config_file doesn't exist)

    Returns:
        Configured device instance

    Example:
        >>> from umi.umi_device.dexumi_exoskeleton import DexUmiExoskeleton
        >>> exo = create_umi_device(
        ...     DexUmiExoskeleton,
        ...     device_name="right_dexumi_exoskeleton",
        ...     reader=uart_reader,
        ...     config_file="config/dexumi_exoskeleton/DEXUMI-Right.yaml",
        ... )
    """
    if config_file:
        config_file = os.path.expanduser(str(config_file))

    if config_file and os.path.exists(config_file):
        print(f"Creating {device_class.__name__} from config: {config_file}")
        device = device_class.from_config(
            config_file,
            reader=reader,
            device_name=device_name,
        )
    else:
        if config_file:
            print(f"Config file not found: {config_file}, creating with params")
        device = device_class(
            device_name=device_name,
            reader=reader,
            **params,
        )

    return device


class UmiDeviceInterface(ABC):
    """Abstract base class for UMI devices.

    Provides common functionality for configuration and calibration management.
    Subclasses must implement device-specific methods.

    Attributes:
        _config_keys: List of attribute names to include in config serialization.
            Subclasses should define this as a class variable.
        action_source_key: Name of the recorded (aligned) zarr field that holds
            the device's calibrated value to feed to ``convert_to_robot_actions``
            in offline conversion. Subclasses must set this.
    """

    _config_keys: list[str] = []

    # Discard known calibration provenance fields before construction.
    # Unexpected sensor settings still raise TypeError.
    _config_metadata_keys: tuple[str, ...] = ()

    # Subclasses set this to their calibrated recorded field (e.g.
    # "joint_angles_interp"); see convert_to_robot_actions / action_source_key.
    action_source_key: str | None = None

    def __init__(self) -> None:
        pass

    @classmethod
    def from_config(
        cls: type[T],
        config_path: str | Path,
        reader: Any,
        **kwargs: Any,
    ) -> T:
        """Create device instance from a configuration file.

        The config file should be a YAML file containing all device init
        parameters. Keys named in ``_config_metadata_keys`` (calibration
        provenance) are stripped before construction; any other key the
        constructor does not accept raises TypeError.

        Args:
            config_path: Path to the YAML configuration file
            reader: Shared UartSensorReader instance
            **kwargs: Override values from config

        Returns:
            Configured device instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is not valid YAML
        """
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        config.update(kwargs)
        config["reader"] = reader

        for key in cls._config_metadata_keys:
            config.pop(key, None)

        return cls(**config)

    @classmethod
    def resolve_config_file(cls, uart_device_name: str) -> str | None:
        """Canonical calibration config path for a device name.

        Returns None when the device type has no per-device config convention
        (e.g. UMI grippers, whose configs are role-based, not per-board).
        """
        return None

    @classmethod
    def validate_config_file(cls, config_path: str) -> None:
        """Raise if this calibration must not drive real hardware.

        Default: no restriction.
        """

    def _validate_required_config(self) -> None:
        """Validate that all required config keys are set.

        Raises:
            ValueError: If any required config keys are None
        """
        missing = []
        for key in self._config_keys:
            value = getattr(self, key, None)
            if value is None:
                missing.append(key)

        if missing:
            raise ValueError(
                f"{self.__class__.__name__} missing required configuration: "
                f"{', '.join(missing)}. Provide via config_path or as parameters."
            )

    @abstractmethod
    def get_data_sources(self) -> list[SensorInterface]:
        """Return data sources for the recorder.

        Each data source implements SensorInterface and can be consumed
        independently by a NumericRecorder.

        Returns:
            List of SensorInterface data sources.
        """
        pass

    @staticmethod
    def load_mapping(config_path: str | Path | None) -> dict | None:
        """Load device-specific mapping from a config file.

        Override in subclasses that need mapping (e.g., polynomial maps).
        Returns None by default (identity mapping).

        Args:
            config_path: Path to config YAML, or None.

        Returns:
            Mapping dict or None.
        """
        return None

    @classmethod
    @abstractmethod
    def convert_to_robot_actions(cls, data: np.ndarray, **kwargs) -> np.ndarray:
        """Convert recorded device data to a robot action (offline).

        Works without hardware. Used offline on recorded data (e.g. by a
        training-data converter): the input is the device's calibrated recorded
        value (see ``action_source_key``) and the output is the robot action.

        Offline callers use ``convert_to_robot_actions(data, umi_config=...)``;
        devices that need config-derived parameters (e.g. polynomial / joint
        maps, hand DoF) load them here from that saved device config, and
        devices that don't ignore the extra kwargs via ``**kwargs``.

        Args:
            data: Recorded calibrated data array, shape (T,) or (T, N).

        Returns:
            Robot action array (robot-hand joint positions).
        """
        pass
