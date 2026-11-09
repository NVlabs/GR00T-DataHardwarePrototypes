# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from umi.common.data.frame import Frame


class SensorInterface(ABC):
    def __init__(
        self,
        device_name: str,
        latency: float = None,
        group_suffix: str = "",
    ) -> None:
        self.device_name = device_name
        self.latency = latency
        # Append group_suffix after the split key: device_joint_state and
        # device_force_state retain the same role across sub-streams. Blank adds no
        # suffix.
        self.group_suffix = group_suffix

        self.is_running = False

    @abstractmethod
    def start_streaming(self):
        raise NotImplementedError

    @abstractmethod
    def get_frame(self) -> Frame:
        raise NotImplementedError

    @abstractmethod
    def stop_streaming(self):
        raise NotImplementedError

    def live_streaming(self):
        pass
