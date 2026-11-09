# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from umi.common.hardware_sdk.camera import oak_device
from umi.common.hardware_sdk.camera.oak_device import OAKDevice, OAKStream


class _Transformation:
    def __init__(self, intrinsics):
        self._intrinsics = intrinsics

    def getIntrinsicMatrix(self):
        return self._intrinsics


class _Frame:
    def __init__(self, image, intrinsics):
        self._image = image
        self._transformation = _Transformation(intrinsics)

    def getCvFrame(self):
        return self._image.copy()

    def getTransformation(self):
        return self._transformation


class _Queue:
    def __init__(self, frame=None):
        self._frame = frame

    def tryGet(self):
        frame, self._frame = self._frame, None
        return frame


def _stream(frame, *, fallback=None):
    stream = OAKStream(
        device_name="test_camera",
        camera_resolution=(640, 400),
        fps=30,
        latency=0.0,
        output_queue=_Queue(frame),
        fallback_intrinsics=fallback,
        oak_device=object(),
    )
    stream.start_streaming()
    return stream


def test_stream_uses_effective_intrinsics_from_frame_metadata():
    intrinsics = np.array([[420.0, 0.0, 311.0], [0.0, 421.0, 198.0], [0.0, 0.0, 1.0]])
    frame = _Frame(np.zeros((400, 640, 3), dtype=np.uint8), intrinsics)

    result = _stream(frame).get_frame()

    np.testing.assert_array_equal(result.intrinsics, intrinsics)


def test_stream_keeps_fallback_when_frame_has_no_calibration():
    fallback = np.array([[198.0, 0.0, 320.0], [0.0, 198.0, 196.0], [0.0, 0.0, 1.0]])
    frame = _Frame(np.zeros((400, 640, 3), dtype=np.uint8), np.eye(3))

    result = _stream(frame, fallback=fallback).get_frame()

    np.testing.assert_array_equal(result.intrinsics, fallback)


def test_pipeline_requests_crop_resize_mode(monkeypatch):
    socket = oak_device.dai.CameraBoardSocket.CAM_A

    class _CameraNode:
        def setSensorType(self, _sensor_type):
            pass

        def build(self, _socket):
            return self

        def requestOutput(self, resolution, **kwargs):
            self.resolution = resolution
            self.output_kwargs = kwargs
            return SimpleNamespace(createOutputQueue=lambda **_kwargs: _Queue())

    camera_node = _CameraNode()

    class _Pipeline:
        def __init__(self, _device):
            pass

        def create(self, _node_type):
            return camera_node

    feature = SimpleNamespace(socket=socket, sensorName="UNKNOWN", configs=[])
    fake_device = SimpleNamespace(
        getConnectedCameras=lambda: [socket],
        getConnectedCameraFeatures=lambda: [feature],
    )
    monkeypatch.setattr(oak_device.dai, "Pipeline", _Pipeline)

    device = object.__new__(OAKDevice)
    device.device = fake_device
    device.fps = 30
    device.latency = 0.0
    device.ae_max_exposure_us = None
    device.allow_cropped_output = False
    device.streams = []
    device.build_pipeline(
        [{"name": "camera", "socket": "CAM_A", "resolution": [640, 400]}]
    )

    assert camera_node.resolution == (640, 400)
    assert camera_node.output_kwargs["resizeMode"] == oak_device.dai.ImgResizeMode.CROP


# Intrinsics precedence: per-frame metadata, device EEPROM, fallback table,
# then None. EEPROM supports modules absent from the table.


class _Calib:
    def __init__(self, matrix):
        self._matrix = matrix
        self.asked = None

    def getCameraIntrinsics(self, socket, resizeWidth, resizeHeight):
        self.asked = (socket, resizeWidth, resizeHeight)
        return self._matrix


def _device(calib=None, raises=False):
    def read():
        if raises:
            raise RuntimeError("no EEPROM on this board")
        return calib

    return SimpleNamespace(readCalibration=read)


def test_eeprom_intrinsics_are_requested_at_the_output_resolution():
    """K must be scaled to what the pipeline actually emits, not the sensor
    mode -- fx off by 10% costs ~29 mm of board pose."""
    calib = _Calib(
        np.array([[466.0, 0.0, 320.0], [0.0, 466.0, 240.0], [0.0, 0.0, 1.0]])
    )
    K = oak_device.eeprom_intrinsics(_device(calib), "CAM_A", (640, 480))
    assert calib.asked == ("CAM_A", 640, 480)
    assert K[0, 0] == 466.0


def test_eeprom_intrinsics_reject_an_uncalibrated_identity_block():
    """An unflashed block reads as identity. Returning it would hand solvePnP a
    plausible-looking K and silently produce a garbage pose."""
    calib = _Calib(np.eye(3))
    assert oak_device.eeprom_intrinsics(_device(calib), "CAM_A", (640, 480)) is None


def test_eeprom_intrinsics_degrade_to_none_rather_than_raising():
    """Devices without a calibration block throw; the cascade must fall through
    to the table instead of taking down the recorder."""
    assert (
        oak_device.eeprom_intrinsics(_device(raises=True), "CAM_A", (640, 480)) is None
    )
    assert oak_device.eeprom_intrinsics(None, "CAM_A", (640, 480)) is None
    assert (
        oak_device.eeprom_intrinsics(_device(_Calib(np.eye(3))), "CAM_A", None) is None
    )


def test_eeprom_intrinsics_never_uses_the_fabricated_default():
    """readCalibrationOrDefault() invents a K for uncalibrated devices. Reaching
    for it would trade a loud failure for a silent ~100 mm error, so a device
    offering both must be asked only for the real one."""
    touched = []
    device = SimpleNamespace(
        readCalibration=lambda: (touched.append("real"), _Calib(np.eye(3)))[1],
        readCalibrationOrDefault=lambda: (
            touched.append("default"),
            _Calib(np.array([[640.0, 0, 320.0], [0, 640.0, 240.0], [0, 0, 1.0]])),
        )[1],
    )
    assert oak_device.eeprom_intrinsics(device, "CAM_A", (640, 480)) is None
    assert touched == ["real"]
