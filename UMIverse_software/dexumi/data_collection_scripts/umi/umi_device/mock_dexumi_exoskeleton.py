# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock DexUMI exoskeleton — a no-hardware stand-in for ``DexUmiExoskeleton``.

It subclasses the real device so it reuses the *exact* encoder + tactile frame
classes (same recorded zarr group names and field names — ``joint_angles`` /
``raw_encoder_angles`` and the ``tactile_{id}_*`` channels), but overrides
``get_data_sources()`` to emit smooth, continuously-varying synthetic data
instead of reading a UART. The recorded data therefore looks like a real DexUMI
recording (right schema), so alignment and downstream tools see the same
structure as the real rig.

Construct it exactly like the real device — it inherits ``from_config`` — but
pass ``reader=None`` (there is no serial link). Wire it into a recording config
with ``type: mock_dexumi_exoskeleton`` (see config/data_recording/dexumi/
mock_bimanual_dexumi_config.yaml).
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from umi.common.hardware_sdk.frame_source import FrameSource
from umi.umi_device.dexumi_exoskeleton import DexUmiExoskeleton

# Frame metadata fields that are never filled with synthetic sensor values.
_META_FIELDS = ("capture_time", "receive_time", "frame_source")


class MockDexUmiExoskeleton(DexUmiExoskeleton):
    """Synthetic DexUMI exoskeleton (encoder + tactile) with no hardware."""

    @classmethod
    def validate_config_file(cls, config_path: str) -> None:
        """No restriction: placeholder calibrations are the mock's purpose."""

    def get_data_sources(self) -> list[FrameSource]:
        """Return encoder (+ tactile) sources backed by synthetic producers.

        Mirrors ``DexUmiExoskeleton.get_data_sources`` (same group names and
        frame classes), but each source is a pull-mode ``FrameSource`` whose
        producer fabricates a frame — no UART read, no underlying sensor stream.
        """
        self._setup_mock_signals()

        sources: list[FrameSource] = [
            FrameSource(
                device_name=f"{self.device_name}_encoder",
                frame_class=self.encoder_frame_class,
                producer=self._mock_encoder_frame,
                latency=self.latency,
            )
        ]
        if self.tactile is not None:
            sources.append(
                FrameSource(
                    device_name=f"{self.device_name}_tactile",
                    frame_class=self.tactile.frame_class,
                    producer=self._mock_tactile_frame,
                    latency=self.latency,
                )
            )
        return sources

    # ------------------------------------------------------------------
    # Synthetic signal generation
    # ------------------------------------------------------------------
    def _setup_mock_signals(self) -> None:
        """Pick randomized-per-run sinusoid params for smooth, moving signals."""
        rng = np.random.default_rng()
        self._mock_t0: float | None = None

        n = len(self.encoder_sensor_ids)
        if self.zero_position_angle is None:
            # A design template has no measured zeros: drift around 0 degrees.
            self.zero_position_angle = np.zeros(n)
        self._enc_amp = rng.uniform(2.0, 8.0, size=n)  # degrees
        self._enc_freq = rng.uniform(0.1, 0.5, size=n)  # Hz
        self._enc_phase = rng.uniform(0.0, 2 * np.pi, size=n)

        self._tac_fields = (
            [
                f.name
                for f in dataclasses.fields(self.tactile.frame_class)
                if f.name not in _META_FIELDS
            ]
            if self.tactile is not None
            else []
        )
        k = len(self._tac_fields)
        # Tactile force uses millinewtons. Keep offset_min greater than amp_max so
        # synthetic contact readings stay strictly positive, and well above the
        # tactile conversion's force deadband (robot_action.deadband_n, 0.1 N =
        # 100 mN): weaker mock readings would give all-zero joint torques.
        self._tac_offset = rng.uniform(600.0, 1500.0, size=k)
        self._tac_amp = rng.uniform(100.0, 500.0, size=k)
        self._tac_freq = rng.uniform(0.1, 0.6, size=k)
        self._tac_phase = rng.uniform(0.0, 2 * np.pi, size=k)

    def _tick(self) -> tuple[float, float, float]:
        """Return (elapsed_seconds, capture_time, receive_time) for this frame."""
        now = time.monotonic()
        if self._mock_t0 is None:
            self._mock_t0 = now
        latency = self.latency or 0.0
        return now - self._mock_t0, now - latency, now

    def _mock_encoder_frame(self):
        """Synthetic encoder frame: raw angles drift around the zero pose, and
        joint_angles is the *real* signed-offset transform of them (self-consistent
        with the device's calibration)."""
        t, capture_time, receive_time = self._tick()
        n = len(self.encoder_sensor_ids)
        base = np.asarray(self.zero_position_angle, dtype=np.float64)[:n]
        raw = base + self._enc_amp * np.sin(
            2 * np.pi * self._enc_freq * t + self._enc_phase
        )
        joint = self.get_signed_angle_offset(raw)
        return self.encoder_frame_class(
            joint_angles=joint.astype(np.float64),
            raw_encoder_angles=raw.astype(np.float64),
            capture_time=capture_time,
            receive_time=receive_time,
            frame_source=f"{self.device_name}_encoder",
        )

    def _mock_tactile_frame(self):
        """Synthetic tactile frame: every channel a smooth positive sinusoid."""
        t, capture_time, receive_time = self._tick()
        vals = self._tac_offset + self._tac_amp * np.sin(
            2 * np.pi * self._tac_freq * t + self._tac_phase
        )
        kwargs = {name: float(v) for name, v in zip(self._tac_fields, vals)}
        return self.tactile.frame_class(
            capture_time=capture_time,
            receive_time=receive_time,
            frame_source=f"{self.device_name}_tactile",
            **kwargs,
        )
