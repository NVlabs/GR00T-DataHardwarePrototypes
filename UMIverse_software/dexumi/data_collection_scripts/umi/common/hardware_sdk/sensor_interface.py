# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

from umi.common.data.frame import Frame


class SensorInterface(ABC):
    def __init__(
        self,
        device_name: str,
        latency: float = None,
    ) -> None:
        self.device_name = device_name
        self.latency = latency
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

    # Name of the unexpanded recording source, shared by its encoder/tactile
    # streams. Recording factories set it; standalone sources may leave it None.
    config_source_name: Optional[str] = None

    def identity(self) -> Optional[dict]:
        """Expected physical device behind this source, from its config.

        ``{"kind": str, "name": str | None, "id": str | None}``. None means the
        source has no device identity (unbound), and identity bookkeeping
        (device_binding.yaml, the ``device_identity`` episode attr) skips it.
        """
        return None

    def observed_identity(self) -> Optional[dict]:
        """What the SDK actually opened: ``{"id": str | None, ...extras}``.

        None when the source cannot report it (see ``identity``).
        """
        return None
