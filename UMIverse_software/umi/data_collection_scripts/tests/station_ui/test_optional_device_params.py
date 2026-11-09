# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional-device source blocks come from config/data_recording/
optional_devices.yaml, and the emitted config is byte-for-byte what it is
pinned to.

Two guards:

  * ``test_*_block_renders_exactly_as_before`` pins the serialized text of each
    emitted block. A YAML round-trip that turned ``true`` into ``"true"``,
    ``0.125`` into ``"0.125"``, or a flow ``[720, 960]`` into a block list would
    change the file the recorder reads; a value comparison would miss all three,
    so these assert the exact bytes.
  * ``test_*_block_is_the_one_declared_in_the_data_file`` proves the file is the
    source of truth -- the emitted block's static fields and params equal the
    file's entry -- so optional_devices.yaml is not a dead copy the code ignores.

OAK is the one type whose ``camera_resolution`` is not static: it is scanned from
the device, so the file carries only OAK's static fields and the assembler still
computes the resolution. ``test_oak_block_scans_its_resolution...`` pins that.
"""

import pathlib

import pytest

ruamel = pytest.importorskip("ruamel.yaml")

from station_ui.core import config_assembler as mod  # noqa: E402
from station_ui.core import setup_presets as sp  # noqa: E402
from station_ui.core.config_assembler import (  # noqa: E402
    ConfigAssembler,
    DeviceSpec,
    SessionSpec,
)
from station_ui.core.device_scan import (  # noqa: E402
    HEAD_STREAM_RESOLUTION,
    resolution_for,
)
from station_ui.core.optional_devices import (  # noqa: E402
    OptionalDeviceAssignment,
    type_for_logical_name,
)
from station_ui.core.setup_presets import template_path  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA_FILE = REPO / "config" / "data_recording" / "production" / "optional_devices.yaml"


@pytest.fixture(autouse=True)
def _isolated_output(tmp_path, monkeypatch):
    """Assemble into tmp_path, never into the station state dir."""
    monkeypatch.setattr(mod, "GENERATED_DIR", (str(tmp_path), "generated"))


def _session(**kw):
    base = dict(
        preset_id="bimanual_v4",
        operator_name="Test Operator",
        task_name="plane_assembly_v9",
        session_uid="91e98c89fccb",
        data_dir="/data/raw/plane_assembly_v9/2026-08-03_x_91e98c89fccb",
    )
    base.update(kw)
    return SessionSpec(**base)


def _template_cameras(preset_id="bimanual_v4"):
    """A device per required camera block: OAK streams share their board, a USB
    camera is named by its V4L2 index, and no two blocks name the same device --
    what assemble() demands of a full rig."""
    preset = sp.get_preset(preset_id)
    cfg = ruamel.YAML().load(template_path(preset, REPO).read_text())
    cameras = {}
    for index, source in enumerate(cfg.get("video_sources") or []):
        names = [s["name"] for s in source.get("streams") or [] if s.get("name")]
        if not names and source.get("name"):
            names = [source["name"]]
        if source.get("type") == "uvc_camera":
            device = str(index)
        else:
            device = f"fixture-device-{index}"
        for name in names:
            if type_for_logical_name(name) is None:
                cameras[name] = device
    return cameras


def _devices(preset_id="bimanual_v4", **kw):
    base = dict(
        grippers={"left": "UMI-V4-0001", "right": "UMI-V4-0002"},
        required_cameras=_template_cameras(preset_id),
        optional=(),
    )
    base.update(kw)
    return DeviceSpec(**base)


def _load(path):
    return ruamel.YAML().load(path.read_text())


def _data_file():
    return ruamel.YAML().load(DATA_FILE.read_text())


_HEAD_CAMERAS = (
    OptionalDeviceAssignment("head_oak", "head_oak_camera", "oak-W97-51", "head"),
    OptionalDeviceAssignment(
        "head_iphone", "head_iphone_camera", "iphone16p-ry", "neck"
    ),
)


def _emitted_block(logical_name, type_id, device, placement=""):
    """The emitted source block for one optional camera."""
    devices = _devices(
        optional=(OptionalDeviceAssignment(type_id, logical_name, device, placement),)
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    return next(s for s in cfg["video_sources"] if s.get("name") == logical_name)


# Assert emitted blocks from real assembled rigs exactly, including YAML formatting.

# Head OAK unscanned: 640x480 (device_scan.HEAD_STREAM_RESOLUTION), not the
# wrist size.
_OAK_TEXT = """\
  - type: oak_camera
    name: head_oak_camera
    record_fps: 30
    stream_fps: 45
    frame_type: RGBCameraFrame
    latency: 0.0875
    params:
      oak_device_name: "oak-W97-51"
      camera_resolution: [640, 480]
      fps: 30
      ae_max_exposure_us: 22222
      enable_color: true
      enable_mono_cameras: false
    placement: head
"""

_IPHONE_TEXT = """\
  - type: iphone_camera
    name: head_iphone_camera
    record_fps: 30
    stream_fps: 90
    frame_type: RGBDCameraFrame
    latency: 0.125
    params:
      iphone_device_name: "iphone16p-ry"
      camera_resolution: [720, 960]
    placement: neck
"""


@pytest.mark.parametrize(
    "expected",
    [_OAK_TEXT, _IPHONE_TEXT],
    ids=["oak", "iphone"],
)
def test_each_optional_block_renders_exactly_as_before(expected):
    """bimanual_v4 + a head OAK and a head iPhone: every block is byte-identical
    to the pinned text. Text, not values: only text catches a bool serialized as
    a string or a flow list broken onto separate lines."""
    out = ConfigAssembler(REPO).assemble(_session(), _devices(optional=_HEAD_CAMERAS))
    text = out.read_text()
    assert expected in text


# Changing the optional-source data file must change emitted static fields and params.

_SOURCE_FIELDS = ("type", "record_fps", "stream_fps", "frame_type", "latency")


def test_iphone_block_is_the_one_declared_in_the_data_file():
    spec = _data_file()["iphone_camera"]
    block = _emitted_block("head_iphone_camera", "head_iphone", "iphone16p-ry")
    for key in _SOURCE_FIELDS:
        assert block[key] == spec[key], key
    # params equal the file's, plus the device identifier injected separately --
    # which the file must not carry, or it would double-write the operator's pick.
    assert "iphone_device_name" not in spec["params"]
    emitted = dict(block["params"])
    assert emitted.pop("iphone_device_name") == "iphone16p-ry"
    assert emitted == dict(spec["params"])


def test_oak_block_takes_static_fields_from_the_file_but_scans_its_resolution():
    """OAK is the exception the file cannot hold: camera_resolution depends on
    the scanned module (a 4:3 IMX378 needs 640x480, not the 16:10 default), so
    the file carries only OAK's static fields and the assembler computes the
    resolution. The static fields still come from the file."""
    spec = _data_file()["oak_camera"]
    block = _emitted_block("head_oak_camera", "head_oak", "oak-W97-51")

    assert block["type"] == spec["type"] == "oak_camera"
    assert block["frame_type"] == spec["frame_type"]
    assert block["latency"] == spec["latency"]
    # OAK states its rates like every other optional camera, rather than
    # inheriting `recording.record_fps`: otherwise lowering the global would
    # move the head camera, and the head camera could not be set on its own.
    assert block["record_fps"] == spec["record_fps"] == 30
    assert block["stream_fps"] == spec["stream_fps"] == 45

    params = dict(block["params"])
    assert params.pop("oak_device_name") == "oak-W97-51"
    # The resolution is scanned, never frozen in the file. Unscanned (as here:
    # no scan backs this assembler) the head falls back to its own 4:3 size,
    # not the wrist default -- see device_scan.HEAD_STREAM_RESOLUTION.
    assert "camera_resolution" not in spec["params"]
    scanned, warning = resolution_for(
        "oak-W97-51", "CAM_A", default=HEAD_STREAM_RESOLUTION
    )
    assert warning is not None and list(scanned) == [640, 480]
    assert list(params.pop("camera_resolution")) == [640, 480]
    # Everything else is the file's static OAK params.
    assert params == dict(spec["params"])


def test_the_bottom_oak_block_is_the_head_oak_block_under_its_own_name():
    """One `oak_camera` spec serves both fixed OAKs. The bottom camera differs
    from the head only in the name it is written under and where it stands, so
    the recorder, the data check and the uploader see nothing new."""
    head = _emitted_block("head_oak_camera", "head_oak", "oak-W97-51", "head")
    bottom = _emitted_block("bottom_oak_camera", "bottom_oak", "oak-W97-51", "table")

    def body(block):
        return {k: v for k, v in dict(block).items() if k not in ("name", "placement")}

    assert body(head) == body(bottom)
    assert bottom["name"] == "bottom_oak_camera"
    assert bottom["placement"] == "table"


def test_the_data_file_holds_no_device_identifier():
    """A *_device_name in the file would be written twice -- once from the file,
    once by _set_device_identifier -- and the two could name different cameras."""
    data = _data_file()
    for source_type, entry in data.items():
        params = entry.get("params") or {}
        offenders = [k for k in params if k.endswith(("_device_name", "_device_id"))]
        assert not offenders, (source_type, offenders)
