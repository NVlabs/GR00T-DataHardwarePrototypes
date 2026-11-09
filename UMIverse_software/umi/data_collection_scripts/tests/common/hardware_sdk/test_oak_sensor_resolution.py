# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""select_sensor_resolution refuses an output that would crop the sensor.

Asking a 4:3 IMX378 head camera for 640x400 makes the ISP crop the image. A
warning that stops nothing lets whole sessions record that crop, so a mismatch
is an error at start-up unless the config says, in so many words, that
cropping is wanted.

No hardware: the device is a duck-typed fake carrying the one method the
selector reads, `getConnectedCameraFeatures`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from umi.common.hardware_sdk.camera.oak_device import select_sensor_resolution


def _feature(socket, sensor, width, height, modes):
    return SimpleNamespace(
        socket=socket,
        sensorName=sensor,
        width=width,
        height=height,
        configs=[
            SimpleNamespace(width=w, height=h, maxFps=fps, type="COLOR")
            for w, h, fps in modes
        ],
    )


class _Device:
    def __init__(self, *features):
        self._features = list(features)

    def getConnectedCameraFeatures(self):
        return self._features


# An OAK-D S2 head camera: 4:3 IMX378 on CAM_A.
IMX378 = _feature(
    "CAM_A",
    "IMX378",
    4056,
    3040,
    [(4056, 3040, 30), (2024, 1520, 85), (1920, 1080, 60)],
)
# A wrist board: 16:10 AR0234.
AR0234 = _feature("CAM_B", "AR0234", 1920, 1200, [(1920, 1200, 60), (1280, 800, 120)])


def test_a_16_10_output_on_a_4_3_sensor_is_refused_with_the_right_size_named():
    with pytest.raises(RuntimeError) as exc:
        select_sensor_resolution(_Device(IMX378), "CAM_A", (640, 400), 30.0)
    msg = str(exc.value)
    assert "IMX378" in msg and "640x400" in msg
    assert "640x480" in msg  # the fix, in the message
    assert "allow_cropped_output" in msg  # the deliberate escape hatch, named


def test_the_native_aspect_is_accepted_and_keeps_the_preferred_full_fov_mode():
    assert select_sensor_resolution(_Device(IMX378), "CAM_A", (640, 480), 30.0) == (
        2024,
        1520,
    )
    assert select_sensor_resolution(_Device(AR0234), "CAM_B", (640, 400), 30.0) == (
        1920,
        1200,
    )


def test_cropping_on_purpose_is_allowed_only_when_the_config_says_so(capsys):
    assert select_sensor_resolution(
        _Device(IMX378), "CAM_A", (640, 400), 30.0, allow_cropped_output=True
    ) == (2024, 1520)
    assert "allow_cropped_output is set" in capsys.readouterr().out


def test_an_unknown_sensor_still_defers_to_depthai_and_fps_shortfalls_still_raise_their_own_error():
    assert select_sensor_resolution(_Device(AR0234), "CAM_A", (640, 480), 30.0) is None
    with pytest.raises(RuntimeError, match="can sustain"):
        select_sensor_resolution(_Device(IMX378), "CAM_A", (640, 480), 120.0)


# ---------------------------------------------- strict on the recording path only


def test_the_recording_path_builds_oak_sources_strict_unless_the_config_opts_in(
    monkeypatch,
):
    """Tools, viewers and calibration keep the permissive class default; the
    recorder passes False, so a cropped request stops a session but not a bench
    script. A source that means to crop says `allow_cropped_output: true`."""
    from umi.common.hardware_sdk.camera import oak_camera, oak_device
    from umi.common.utility import recording_utils

    seen = {}

    class FakeCamera:
        def __init__(self, **kw):
            seen["camera"] = kw

    class FakeDevice:
        def __init__(self, **kw):
            seen["device"] = kw

    monkeypatch.setattr(oak_camera, "OAKCamera", FakeCamera)
    monkeypatch.setattr(oak_device, "OAKDevice", FakeDevice)
    recording_utils._oak_device_cache.clear()

    recording_utils.create_video_source(
        {
            "type": "oak_camera",
            "name": "head",
            "params": {"oak_device_name": "oak-head", "camera_resolution": [640, 480]},
        }
    )
    assert seen["camera"]["allow_cropped_output"] is False
    recording_utils.create_video_source(
        {
            "type": "oak_camera",
            "name": "head",
            "params": {"oak_device_name": "oak-head", "allow_cropped_output": True},
        }
    )
    assert seen["camera"]["allow_cropped_output"] is True

    recording_utils._get_or_create_oak_device({"oak_device_name": "oak-FFC-01"})
    assert seen["device"]["allow_cropped_output"] is False
    recording_utils._oak_device_cache.clear()
    recording_utils._get_or_create_oak_device(
        {"oak_device_name": "oak-FFC-02", "allow_cropped_output": True}
    )
    assert seen["device"]["allow_cropped_output"] is True


def test_the_classes_default_permissive_for_viewers_and_bench_tools():
    import inspect

    from umi.common.hardware_sdk.camera import oak_camera, oak_device

    assert (
        inspect.signature(oak_camera.OAKCamera.__init__)
        .parameters["allow_cropped_output"]
        .default
        is True
    )
    assert (
        inspect.signature(oak_device.OAKDevice.__init__)
        .parameters["allow_cropped_output"]
        .default
        is True
    )
