# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEAR Neck control: three SteadyWin GIM3505-9 actuators on GDK drives over CAN.

from gear_neck import Neck

with Neck() as neck:
    print(neck.angles())
    neck.enable()
    neck.move_to(yaw=20, pitch=10, roll=0)
"""

# The settings loader is gear_neck.load_config.load_config; the package does not
# re-export it, as the name would hide the module of the same name.
from .load_config import ConfigError, NeckConfig
from .neck import Angles, DriveInfo, Neck, NeckError

__all__ = ["Angles", "ConfigError", "DriveInfo", "Neck", "NeckConfig", "NeckError"]
