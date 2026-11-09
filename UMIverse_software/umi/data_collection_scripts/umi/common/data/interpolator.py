# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interpolator classes for time-series data alignment.

Provides interpolators for different data types:
- LinearInterpolator: For 1D/2D numeric arrays
- PoseInterpolator: For 4x4 homogeneous transformation matrices (uses Slerp for rotations)
"""

import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy import interpolate
from scipy.spatial import transform as st
from scipy.spatial.transform import Rotation

from umi.common.utility.pose import convert_homogeneous_matrix

logger = logging.getLogger(__name__)


class BaseInterpolator(ABC):
    """Base class for interpolators with shared timestamp cleaning logic."""

    def __init__(self, timestamps: np.ndarray, values: np.ndarray):
        """Initialize the interpolator.

        Args:
            timestamps: Array of timestamps.
            values: Array of values to interpolate.
        """
        timestamps = np.asarray(timestamps)
        values = np.asarray(values)

        timestamps, values = self._clean_timestamps(timestamps, values)

        self._build_interpolator(timestamps, values)

    def _find_non_monotonic_indices(self, timestamps: np.ndarray) -> List[int]:
        """Find indices where timestamps are not monotonically increasing.

        Args:
            timestamps: Array of timestamps.

        Returns:
            List of indices to drop.
        """
        return [
            idx
            for idx in range(1, len(timestamps))
            if timestamps[idx] <= timestamps[idx - 1]
        ]

    def _clean_timestamps(
        self, timestamps: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove non-monotonic timestamps and corresponding values.

        Args:
            timestamps: Array of timestamps.
            values: Array of values.

        Returns:
            Tuple of (cleaned_timestamps, cleaned_values).
        """
        drop_indices = self._find_non_monotonic_indices(timestamps)

        if drop_indices:
            logger.warning(
                "Found %d non-monotonic timestamp(s) that will be removed. Indices: %s",
                len(drop_indices),
                drop_indices,
            )
            timestamps = np.delete(timestamps, drop_indices)
            values = np.delete(values, drop_indices, axis=0)

        return timestamps, values

    @abstractmethod
    def _build_interpolator(self, timestamps: np.ndarray, values: np.ndarray) -> None:
        """Build the internal interpolator. Implemented by subclasses."""
        pass

    @abstractmethod
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        """Interpolate values at the given timestamps."""
        pass


class LinearInterpolator(BaseInterpolator):
    """Interpolates 1D or 2D arrays using linear interpolation."""

    def _build_interpolator(self, timestamps: np.ndarray, values: np.ndarray) -> None:
        """Build scipy linear interpolator."""
        self._interp = interpolate.interp1d(timestamps, values, kind="linear", axis=0)

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        """Interpolate values at the given timestamps.

        Args:
            ts: Array of timestamps to interpolate at.

        Returns:
            Interpolated values.
        """
        return self._interp(ts)


class PoseInterpolator(BaseInterpolator):
    """Interpolates 4x4 homogeneous transformation matrices.

    Uses linear interpolation for positions and Slerp for rotations.
    """

    def _clean_timestamps(
        self, timestamps: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Override to handle pose-specific cleaning."""
        drop_indices = self._find_non_monotonic_indices(timestamps)

        if drop_indices:
            logger.warning(
                "Found %d non-monotonic timestamp(s) that will be removed. Indices: %s",
                len(drop_indices),
                drop_indices,
            )
            timestamps = np.delete(timestamps, drop_indices)
            # For poses, delete along axis 0 (values is 3D)
            values = np.delete(values, drop_indices, axis=0)

        return timestamps, values

    def _build_interpolator(self, timestamps: np.ndarray, values: np.ndarray) -> None:
        """Build position and rotation interpolators."""
        positions = values[:, :3, 3]
        rotations = Rotation.from_matrix(values[:, :3, :3])

        self._pos_interp = interpolate.interp1d(
            timestamps, positions, kind="linear", axis=0
        )
        self._rot_interp = st.Slerp(timestamps, rotations)

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        """Interpolate poses at the given timestamps.

        Args:
            ts: Array of timestamps to interpolate at.

        Returns:
            Array of 4x4 homogeneous transformation matrices.
        """
        positions = self._pos_interp(ts)
        rotations = self._rot_interp(ts)
        rotation_matrix = rotations.as_matrix()

        homogeneous_matrix = np.array(
            [
                convert_homogeneous_matrix(rotation_matrix=rot, translation=pos)
                for rot, pos in zip(rotation_matrix, positions)
            ]
        )
        return homogeneous_matrix
