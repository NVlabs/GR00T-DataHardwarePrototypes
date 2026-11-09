# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gripper data-collection pipeline version.

Written to ``meta.gripper_pipeline`` in every recording config and read back by
the recorder and the config editor. V4 is the only version this repository
records; a value it does not name still opens (see :meth:`GripperPipeline.parse`)
and is reported by the recorder rather than refused.
"""

from enum import Enum
from typing import Optional


class GripperPipeline(str, Enum):
    """A pipeline version. ``str`` subclass so ``pipeline == "v4"`` holds and
    YAML round-trips as the plain string."""

    # The flexure gripper: 15 mm pinion, ~148 deg of travel, one OAK wrist
    # camera on CAM_B and a USB trigger camera. See `gear_radius_mm` and
    # `open_travel_band_deg` below for the geometry it records with.
    V4 = "v4"

    def __str__(self) -> str:
        return self.value

    @property
    def gear_radius_mm(self) -> float:
        """Pinion pitch radius the encoder angle is converted to mm with.

        The V4 flexure gripper's printed gear is module 2.5 with 12 teeth, pitch
        radius 15.0 mm (CAD), confirmed on a V4 prototype: ~148 deg of travel
        for a ~78 mm opening. The driver's model is 2*pi*r*deg/360*2, so a
        radius that is off by some factor scales every recorded distance by the
        same factor.
        """
        return 15.0

    @property
    def open_travel_band_deg(self) -> tuple[float, float]:
        """Encoder degrees a full open may land in during calibration.

        120-175: provisional, from one prototype that opened ~148 deg. The
        floor rejects a two-thirds open (~100 deg); the ceiling stays under the
        180 deg wrap of `get_signed_angle_offset`, past which an opening reads
        as closed. Narrow it once more grippers have been measured.
        """
        return (120.0, 175.0)

    @classmethod
    def default(cls) -> "GripperPipeline":
        return cls.V4

    @classmethod
    def values(cls) -> list:
        return [member.value for member in cls]

    @classmethod
    def parse(
        cls,
        value,
        default: Optional["GripperPipeline"] = None,
    ) -> "GripperPipeline":
        """Coerce a config value to a member, falling back rather than raising.

        A config carrying an unknown version must still open — refusing to load
        it would strand an operator on a station they cannot use. Callers that
        want to warn should use :meth:`is_known` first.
        """
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        for member in cls:
            if member.value == text:
                return member
        return cls.default() if default is None else default

    @classmethod
    def is_known(cls, value) -> bool:
        """Whether ``value`` names a real version, for warning on the odd one."""
        if isinstance(value, cls):
            return True
        return str(value or "").strip().lower() in set(cls.values())
