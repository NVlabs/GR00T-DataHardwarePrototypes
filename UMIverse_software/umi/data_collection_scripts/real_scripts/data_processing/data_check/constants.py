# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants shared across data check modules.

There is deliberately no canonical list of expected device names here: expected
names are derived per session from its own recording config (see
`device_names.expected_device_names`), because optional head/bottom cameras are
per-session and a fixed role list quarantines any rig it does not name.
"""

import os

from umi.common.recording.constants import PENDING_SUFFIX

__all__ = [
    "PENDING_SUFFIX",
    "DEFAULT_CHECK_CONFIG_PATH",
]


DEFAULT_CHECK_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "config",
    "data_check",
    "data_check_config.yaml",
)
