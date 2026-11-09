# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-episode checks for UMI device hardware."""

from __future__ import annotations

from umi.common.recording.pre_episode_checks import (
    PreEpisodeContext,
    register,
)
from umi.common.utility.recording_utils import assert_all_uart_readers_connected


@register("umi_device.uart_connected")
class UmiDeviceUartConnectedCheck:
    """Verify all cached UMI UART readers are still attached to their devices."""

    def run(self, ctx: PreEpisodeContext) -> tuple[bool, str]:
        try:
            assert_all_uart_readers_connected()
        except AssertionError as e:
            return False, str(e)
        return True, "all UART readers connected"
