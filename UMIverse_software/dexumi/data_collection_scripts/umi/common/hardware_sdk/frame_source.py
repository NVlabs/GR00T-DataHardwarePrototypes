# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make a callable recordable as a ``SensorInterface`` stream.

The recorder records anything implementing ``SensorInterface``
(``start_streaming`` / ``stop_streaming`` / ``get_frame``). ``FrameSource``
adapts things that are not sensors but still need recording into that
contract: a producer callable that returns a ``Frame`` per poll (e.g. the mock
DexUMI exoskeleton's synthetic frames).
"""

from typing import Callable, Optional, Type

from umi.common.data.frame import Frame
from umi.common.hardware_sdk.sensor_interface import SensorInterface


class FrameSource(SensorInterface):
    """A ``SensorInterface`` whose ``get_frame()`` calls ``producer()``::

        FrameSource("right_exo", EncoderFrame, producer=read_encoder_frame)

    The producer stamps an authoritative ``receive_time`` on its frames (the
    recorder dedups on ``receive_time > last``), so frames pass through
    untouched; only a missing ``frame_source`` is filled in.

    Args:
        device_name: Recorder source name (becomes the zarr group name).
        frame_class: The ``Frame`` subclass this source emits. Carried on the
            instance so a recorder can be built from ``source.frame_class``.
        producer: Callable returning a ``Frame`` or ``None``.
        latency: Optional sensor latency (seconds), forwarded to SensorInterface.
        start_fn / stop_fn: Optional lifecycle hooks (default no-op — a
            producer backed by hardware already runs its own background
            reader).
    """

    def __init__(
        self,
        device_name: str,
        frame_class: Type[Frame],
        *,
        producer: Callable[[], Optional[Frame]],
        latency: Optional[float] = None,
        start_fn: Optional[Callable[[], None]] = None,
        stop_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(device_name=device_name, latency=latency)
        self.frame_class = frame_class
        self._producer = producer
        self._start_fn = start_fn
        self._stop_fn = stop_fn

    def start_streaming(self) -> None:
        self.is_running = True
        if self._start_fn is not None:
            self._start_fn()

    def stop_streaming(self) -> None:
        self.is_running = False
        if self._stop_fn is not None:
            self._stop_fn()

    def get_frame(self) -> Optional[Frame]:
        frame = self._producer()
        if frame is None:
            return None
        if frame.frame_source is None:
            frame.frame_source = self.device_name
        return frame
