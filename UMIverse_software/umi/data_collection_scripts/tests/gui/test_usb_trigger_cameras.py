# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The V4 rig records a USB trigger camera per arm.

The trigger camera is a plain USB (UVC) module, not a stream on an OAK board:
the operator picks a V4L2 index and `UVCCamera` opens it with
`cv2.VideoCapture(index, CAP_V4L2)`, so the index has to reach the config as an
int. There is no UVC registry, so the device box is free text, and every device
box, verification role and preview tile comes from the preset's template.
"""

import pathlib

import pytest

ruamel = pytest.importorskip("ruamel.yaml")

from station_ui.calibration import verification_roles  # noqa: E402
from station_ui.core import config_assembler as mod  # noqa: E402
from station_ui.core.config_assembler import (  # noqa: E402
    ConfigAssembler,
    DeviceSpec,
    SessionSpec,
)

REPO = pathlib.Path(__file__).resolve().parents[2]

_GRIPPERS = {"left": "UMI-V4-0001", "right": "UMI-V4-0002"}

# What the operator typed into the two trigger boxes: /dev/video4 and
# /dev/video6, the second capture node of each USB camera on a two-cam bench.
LEFT_INDEX = 4
RIGHT_INDEX = 6


@pytest.fixture(autouse=True)
def _isolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "GENERATED_DIR", (str(tmp_path), "generated"))


def _session(**kw):
    base = dict(
        preset_id="bimanual_v4",
        operator_name="Test Operator",
        task_name="plane_assembly_v9",
        session_uid="91e98c89fccb",
        data_dir="~/Dev/UMI/local_data/2026-08-03_x_91e98c89fccb",
    )
    base.update(kw)
    return SessionSpec(**base)


def _blocks(preset_id):
    return {
        block.key: block
        for block in ConfigAssembler(REPO).template_requirements(preset_id).blocks
    }


def _v4_rig(**override):
    """Every camera the V4 iPhone template demands, one device per block."""
    cameras = {
        "left_tracking_camera": "iphone15p-09",
        "right_tracking_camera": "iphone15p-33",
        "left_umi_camera": "oak-FFC-08",
        "right_umi_camera": "oak-FFC-22",
        "left_trigger_camera": str(LEFT_INDEX),
        "right_trigger_camera": str(RIGHT_INDEX),
    }
    cameras.update(override)
    return DeviceSpec(grippers=_GRIPPERS, required_cameras=cameras, optional=())


def _assemble():
    out = ConfigAssembler(REPO).assemble(_session(), _v4_rig())
    return ruamel.YAML().load(out.read_text())


def _source(cfg, name):
    for source in cfg.get("video_sources") or []:
        if source.get("name") == name:
            return source
    return None


@pytest.mark.parametrize("preset_id", ["bimanual_v4", "bimanual_v4_vive"])
def test_the_form_asks_for_a_trigger_camera_per_arm(preset_id):
    """One device box per arm, of the kind that has no registry to pick from."""
    blocks = _blocks(preset_id)
    for name in ("left_trigger_camera", "right_trigger_camera"):
        assert name in blocks, f"{name} is not a device the {preset_id} form asks for"
        assert blocks[name].source_type == "uvc_camera"
        assert blocks[name].device_kind == "uvc"
        # Its own device, not a second stream on the arm's OAK.
        assert blocks[name].stream_names == (name,)


def test_the_operators_index_reaches_the_recorder_as_an_int():
    """`UVCCamera.start` does `cv2.VideoCapture(self.camera_index, CAP_V4L2)`
    and `f"/dev/video{self.camera_index}"`. A quoted "4" is a filename to the
    first and a working path to the second, so the failure is a camera that
    opens on some stations and not others."""
    cfg = _assemble()
    for name, index in (
        ("left_trigger_camera", LEFT_INDEX),
        ("right_trigger_camera", RIGHT_INDEX),
    ):
        source = _source(cfg, name)
        assert source is not None, f"{name} is missing from the assembled config"
        assert source["type"] == "uvc_camera"
        assert source["params"]["camera_index"] == index
        assert isinstance(source["params"]["camera_index"], int)


def test_the_index_may_be_typed_the_way_v4l2_ctl_prints_it():
    """`uvc_camera.get_connected_cameras` parses `/dev/videoN` out of v4l2-ctl
    and that is what an operator reads off the bench, so the box takes it."""
    cfg = ruamel.YAML().load(
        ConfigAssembler(REPO)
        .assemble(_session(), _v4_rig(left_trigger_camera="/dev/video4"))
        .read_text()
    )
    assert _source(cfg, "left_trigger_camera")["params"]["camera_index"] == 4


def test_a_trigger_box_that_is_not_an_index_is_refused_by_name():
    """There is no UVC registry, so the box is free text and a device name is
    the plausible wrong answer. It has to be refused naming the camera --
    `_errors_for` matches the message against each block's stream names to
    decide which row to light up, and an unnamed one renders as a bare banner.
    """
    with pytest.raises(ValueError) as excinfo:
        ConfigAssembler(REPO).assemble(
            _session(), _v4_rig(right_trigger_camera="oak-FFC-22")
        )
    assert "right_trigger_camera" in str(excinfo.value)


def test_the_camera_check_verifies_both_trigger_cameras():
    """The check panel is keyed on `camera_roles.ROLE_SOURCE_NAMES` filtered by
    what the config declares, so a template with no trigger block badges the
    step Done with two cameras never looked at."""
    roles = verification_roles(_assemble())
    assert "left_trigger" in roles
    assert "right_trigger" in roles


def test_the_templates_capture_settings_are_kept():
    """Assembly writes the operator's index and leaves the rest of the block as
    the template declares it: the rate the module delivers, the size the data
    check expects, and the latency the aligner subtracts."""
    cfg = _assemble()
    for name in ("left_trigger_camera", "right_trigger_camera"):
        source = _source(cfg, name)
        assert source["record_fps"] == 25
        assert source["stream_fps"] == 30
        assert source["frame_type"] == "RGBCameraFrame"
        assert source["latency"] == 0.0917
        assert list(source["params"]["camera_resolution"]) == [1280, 720]
        assert source["params"]["fps"] == 30
        assert source["params"]["pixel_format"] == "MJPG"
