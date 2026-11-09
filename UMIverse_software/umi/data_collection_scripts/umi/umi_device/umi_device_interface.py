# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

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
        Config file structure (e.g., gripper_config.yaml):
        device_name: umi_gripper
        gear_radius: 26.0
        distance_zero_angle: 180.0
        directions: '+'

        Usage:
        >>> from umi.umi_device.umi_gripper import UmiGripper
        >>> gripper = create_umi_device(
        ...     UmiGripper,
        ...     device_name="gripper",
        ...     reader=uart_reader,
        ...     config_file="~/.config/gripper.yaml",
        ...     gear_radius=26.0,  # fallback if config doesn't exist
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
        _refuse_a_calibration_from_another_gear(device, config_file, params)
    else:
        if config_file:
            print(f"Config file not found: {config_file}, creating with params")
        device = device_class(
            device_name=device_name,
            reader=reader,
            **params,
        )

    return device


def _refuse_a_calibration_from_another_gear(device, config_file, params) -> None:
    """The saved calibration wins over the recording config -- unless they
    describe different hardware.

    ``params`` is what the template declares for this source; the file is what
    the operator calibrated. They are allowed to disagree on nothing physical:
    a V4 preset over a file calibrated with another gear radius would record
    every distance scaled by the wrong gear, silently, for the whole session.
    Refusing here puts the operator back on the calibration step, which rewrites
    the file for the setup actually in use.
    """
    checks = (
        ("gear_radius", "mm gear radius"),
        ("gripper_generation", "gripper generation"),
    )
    for key, what in checks:
        wanted = params.get(key)
        have = getattr(device, key, None)
        if wanted is None or have is None:
            continue
        if key == "gear_radius":
            different = abs(float(wanted) - float(have)) > 1e-6
        else:
            different = str(wanted).strip().lower() != str(have).strip().lower()
        if different:
            raise ValueError(
                f"{config_file} was calibrated with {have} {what}, but this "
                f"recording config declares {wanted}: the calibration belongs to a "
                "different gripper generation. Recalibrate the gripper under the "
                "current setup before recording -- otherwise every distance would "
                "be scaled by the wrong gear."
            )


class UmiDeviceInterface(ABC):
    """Abstract base class for UMI devices.

    Provides common functionality for configuration and calibration management.
    Subclasses must implement device-specific methods.

    Attributes:
        _config_keys: List of attribute names to include in config serialization.
            Subclasses should define this as a class variable.
    """

    _config_keys: list[str] = []

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

        The config file should be a YAML file containing all device init parameters.

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
            config = yaml.safe_load(f)

        config.update(kwargs)
        config["reader"] = reader

        return cls(**config)

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

    # ==================== Calibration Methods ====================

    @abstractmethod
    def interactive_calibrate(self) -> bool:
        """Run interactive calibration routine.

        Returns:
            True if calibration succeeded, False otherwise
        """
        pass

    # ==================== Config Methods ====================

    def get_config(self) -> dict[str, Any]:
        """Get current device configuration as a dictionary.

        Uses _config_keys to determine which attributes to serialize.

        Returns:
            Dictionary containing device configuration
        """
        config = {}
        for key in self._config_keys:
            if hasattr(self, key):
                config[key] = getattr(self, key)
        return config

    def save_config(self, config_path: str | Path) -> None:
        """Save current configuration to a YAML file.

        Args:
            config_path: Path to save the configuration file
        """
        config_path = Path(config_path)

        config = self.get_config()

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Config saved to {config_path}")
