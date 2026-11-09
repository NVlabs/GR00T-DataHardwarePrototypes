# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared --device/--port/--config handling for the exoskeleton example scripts.

``--device`` is an exact exoskeleton name (DEXUMI-Left or DEXUMI-Right).
The calibration config is derived from it via the strict mapping in
``umi.utils.device_manager``, and the UART reader is opened BY NAME, which
resolves the port from the USB serial the calibration records
(``device_serial``) and verifies the connected device at open (plus per-frame
board-id checks while streaming). A hand whose calibration records no board yet
is opened on the single connected board instead (``connected_board_port``).

``--port`` / ``--config`` are explicit bench overrides that win; opening by raw
port skips the name-based identity verification.
"""

import time

import click

from umi.common.hardware_sdk.sensor import EncoderSensor, UartSensorReader
from umi.utils.device_manager import get_exoskeleton_config_path

BAUDRATE = 1000000


def device_options(f):
    """Shared --device/--port/--config click options (stacked bottom-up)."""
    f = click.option(
        "--config",
        "config_override",
        default=None,
        help="Calibration config YAML override",
    )(f)
    f = click.option(
        "--port",
        default=None,
        help="UART port override (bench use; skips name-based identity verification)",
    )(f)
    f = click.option(
        "--device",
        default=None,
        help=(
            "Exo device name, e.g. DEXUMI-Right. Derives the calibration "
            "config and opens the reader by name (port resolved from the USB "
            "serial the calibration records, identity verified at open)."
        ),
    )(f)
    return f


def resolve_config_path(
    device: str | None,
    config_override: str | None,
    require_exists: bool = True,
) -> str:
    """Explicit --config wins; otherwise derive it from the device name."""
    if config_override:
        return config_override
    if not device:
        raise SystemExit("Error: --device is required unless --config is given")
    return get_exoskeleton_config_path(device, require_exists=require_exists)


def connected_board_port() -> str:
    """Port of the single Pico-family board plugged in.

    A board before its first calibration has no ``device_serial`` to open it by
    name. RP2040 USB descriptors cannot tell a DexUMI exo from another Pico
    board, so the rule is one board total. One hand enumerates as several CDC
    ports sharing one USB serial, so boards are counted by distinct serial.
    """
    ports = UartSensorReader.list_connected_devices(filter_pico=True)
    serials = sorted({p["serial_number"] for p in ports if p["serial_number"]})
    if not serials:
        raise SystemExit(
            "No Pico-family board connected. Plug in the hand (and only the "
            "hand), or pass --port."
        )
    if len(serials) > 1:
        lines = "\n".join(
            f"  {p['device']}  serial={p['serial_number']}  {p['description']}"
            for p in ports
        )
        raise SystemExit(
            f"{len(serials)} Pico-family boards connected; cannot tell which one "
            f"is the hand:\n{lines}\nUnplug the others, or pass --port."
        )
    return UartSensorReader.find_port_by_serial(serials[0])


def open_reader(
    device: str | None,
    port: str | None,
    baudrate: int = BAUDRATE,
) -> UartSensorReader:
    """Open (and start) the UART reader — by name when no explicit port is given."""
    if port:
        reader = UartSensorReader(port=port, baudrate=baudrate)
    elif device:
        reader = UartSensorReader(uart_device_name=device, baudrate=baudrate)
    else:
        raise SystemExit("Error: --device is required unless --port is given")
    reader.start()
    return reader


def open_encoder(
    *,
    sensor_ids: list[int],
    device: str | None = None,
    port: str | None = None,
    device_name: str = "exo_encoder",
) -> tuple[UartSensorReader, EncoderSensor]:
    """Open the reader plus a streaming EncoderSensor; waits for first frames."""
    reader = open_reader(device, port)
    encoder = EncoderSensor(
        device_name=device_name,
        reader=reader,
        sensor_ids=sensor_ids,
    )
    encoder.start_streaming()
    time.sleep(0.5)  # let the first frames arrive
    return reader, encoder
