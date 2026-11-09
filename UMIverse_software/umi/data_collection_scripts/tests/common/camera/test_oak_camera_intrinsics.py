# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from umi.common.data.frame import RGBCameraFrame, RGBDCameraFrame
from umi.common.hardware_sdk.camera.oak_camera import OAKCamera


class _Transformation:
    def __init__(self, intrinsics):
        self._intrinsics = intrinsics

    def getIntrinsicMatrix(self):
        return self._intrinsics


class _RGBFrame:
    def __init__(self, image, intrinsics=None, *, transformation_error=False):
        self._image = image
        self._intrinsics = intrinsics
        self._transformation_error = transformation_error

    def getCvFrame(self):
        return self._image.copy()

    def getTransformation(self):
        if self._transformation_error:
            raise RuntimeError("no transformation metadata")
        return _Transformation(self._intrinsics)


class _DepthFrame:
    def __init__(self, image):
        self._image = image

    def getFrame(self):
        return self._image.copy()

    def getTransformation(self):
        raise AssertionError("intrinsics must come from the aligned RGB frame")


class _Queue:
    def __init__(self, value):
        self.value = value

    def tryGet(self):
        value, self.value = self.value, None
        return value


def _camera(queue_name, value, *, fallback=None):
    camera = object.__new__(OAKCamera)
    camera.device_name = "test_camera"
    camera.is_running = True
    camera.pipeline = SimpleNamespace(isRunning=lambda: True)
    camera.device = SimpleNamespace(isPipelineRunning=lambda: True)
    camera.output_queues = {queue_name: _Queue(value)}
    camera.latency = 0.0
    camera.debug_timing = False
    camera.intrinsic_matrix = fallback
    return camera


def test_color_frame_uses_effective_intrinsics_and_updates_cache():
    intrinsics = np.array([[420.0, 0.0, 311.0], [0.0, 421.0, 198.0], [0.0, 0.0, 1.0]])
    rgb = np.zeros((400, 640, 3), dtype=np.uint8)
    camera = _camera("color", _RGBFrame(rgb, intrinsics), fallback=np.eye(3))

    frame = camera.get_frame()

    assert isinstance(frame, RGBCameraFrame)
    np.testing.assert_array_equal(frame.intrinsics, intrinsics)
    np.testing.assert_array_equal(camera.intrinsic_matrix, intrinsics)


def test_synced_depth_uses_rgb_frame_intrinsics():
    intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 501.0, 200.0], [0.0, 0.0, 1.0]])
    rgb = np.zeros((400, 640, 3), dtype=np.uint8)
    depth = np.full((400, 640), 1000, dtype=np.uint16)
    group = {
        "rgb": _RGBFrame(rgb, intrinsics),
        "depth_aligned": _DepthFrame(depth),
    }
    camera = _camera("synced", group)

    frame = camera.get_frame()

    assert isinstance(frame, RGBDCameraFrame)
    np.testing.assert_array_equal(frame.rgb, rgb)
    np.testing.assert_array_equal(frame.depth, depth)
    np.testing.assert_array_equal(frame.intrinsics, intrinsics)
    np.testing.assert_array_equal(camera.intrinsic_matrix, intrinsics)


def test_invalid_metadata_keeps_fallback_then_last_valid_intrinsics():
    fallback = np.array([[198.0, 0.0, 320.0], [0.0, 198.0, 196.0], [0.0, 0.0, 1.0]])
    valid = np.array([[410.0, 0.0, 319.0], [0.0, 411.0, 201.0], [0.0, 0.0, 1.0]])
    rgb = np.zeros((400, 640, 3), dtype=np.uint8)
    camera = _camera("color", _RGBFrame(rgb, np.eye(3)), fallback=fallback)

    first = camera.get_frame()
    np.testing.assert_array_equal(first.intrinsics, fallback)

    camera.output_queues["color"] = _Queue(_RGBFrame(rgb, valid))
    second = camera.get_frame()
    np.testing.assert_array_equal(second.intrinsics, valid)

    camera.output_queues["color"] = _Queue(_RGBFrame(rgb, transformation_error=True))
    third = camera.get_frame()
    np.testing.assert_array_equal(third.intrinsics, valid)


def test_manually_set_intrinsics_are_used_until_valid_metadata_arrives():
    manual = np.array([[300.0, 0.0, 320.0], [0.0, 301.0, 200.0], [0.0, 0.0, 1.0]])
    emitted = np.array([[420.0, 0.0, 318.0], [0.0, 421.0, 199.0], [0.0, 0.0, 1.0]])
    rgb = np.zeros((400, 640, 3), dtype=np.uint8)
    camera = _camera("color", _RGBFrame(rgb, np.eye(3)))
    camera.set_intrinsic_matrix(manual)

    first = camera.get_frame()
    np.testing.assert_array_equal(first.intrinsics, manual)

    camera.output_queues["color"] = _Queue(_RGBFrame(rgb, emitted))
    second = camera.get_frame()
    np.testing.assert_array_equal(second.intrinsics, emitted)


def test_resolution_override_scales_only_returned_intrinsics():
    intrinsics = np.array([[400.0, 0.0, 320.0], [0.0, 410.0, 200.0], [0.0, 0.0, 1.0]])
    rgb = np.zeros((400, 640, 3), dtype=np.uint8)
    camera = _camera("color", _RGBFrame(rgb, intrinsics))

    frame = camera.get_frame(camera_resolution=(320, 240))

    expected = intrinsics.copy()
    expected[0, :] *= 0.5
    expected[1, :] *= 0.6
    assert frame.rgb.shape == (240, 320, 3)
    np.testing.assert_array_equal(frame.intrinsics, expected)
    np.testing.assert_array_equal(camera.intrinsic_matrix, intrinsics)
