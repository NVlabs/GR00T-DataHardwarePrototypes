# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device detection utilities for the station console."""

from umi.utils.device_manager import IPHONE_DEVICES, OAK_DEVICES


def get_available_iphones():
    """Get list of available iPhone device names"""
    return list(IPHONE_DEVICES.keys())


def get_available_oak_cameras():
    """Get list of available OAK camera names"""
    return list(OAK_DEVICES.keys())
