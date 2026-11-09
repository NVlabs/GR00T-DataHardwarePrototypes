# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor hardware SDK components.

This module provides interfaces for various sensor devices that communicate
via UART using the UMI protocol.
"""

from .encoder_sensor import EncoderSensor
from .protocol import (
    SENSOR_TYPE_JOINT_ENCODER,
    SENSOR_TYPE_SENSOR_HEALTH,
    SENSOR_TYPE_TACTILE,
    Packet,
    PacketParser,
    SensorHealth,
    SensorSegment,
    SensorState,
    Tle5012bError,
    parse_sensor_health,
    register_error_name_resolver,
    register_health_extra_parser,
)
from .sensor_device import SensorDevice, SensorFaultError, SensorReading, SensorSpec
from .tactile_sensor import TactileSensor
from .uart_reader import UartSensorReader

__all__ = [
    "PacketParser",
    "Packet",
    "SensorSegment",
    "SensorHealth",
    "SensorState",
    "Tle5012bError",
    "SENSOR_TYPE_JOINT_ENCODER",
    "SENSOR_TYPE_TACTILE",
    "SENSOR_TYPE_SENSOR_HEALTH",
    "parse_sensor_health",
    "register_health_extra_parser",
    "register_error_name_resolver",
    "UartSensorReader",
    "SensorDevice",
    "SensorSpec",
    "SensorReading",
    "SensorFaultError",
    "EncoderSensor",
    "TactileSensor",
]
