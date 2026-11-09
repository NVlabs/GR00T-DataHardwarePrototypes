# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

ruamel = pytest.importorskip("ruamel.yaml")

import pathlib  # noqa: E402
import shutil  # noqa: E402

from station_ui.core import config_assembler as mod  # noqa: E402
from station_ui.core import setup_presets as sp  # noqa: E402
from station_ui.core.config_assembler import (  # noqa: E402
    ConfigAssembler,
    DeviceSpec,
    SessionSpec,
)
from station_ui.core.optional_devices import (  # noqa: E402
    OptionalDeviceAssignment,
    type_for_logical_name,
)
from station_ui.core.setup_presets import template_path  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402
from umi.common.utility.gripper_pipeline import GripperPipeline  # noqa: E402
from umi.common.utility.reference_camera import video_source_names  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO / "config" / "data_recording"


def _session(**kw):
    base = dict(
        preset_id="bimanual_v4",
        operator_name="Test Operator",
        task_name="plane_assembly_v9",
        session_uid="91e98c89fccb",
        data_dir="output/raw/plane_assembly_v9/2026-08-03_x_91e98c89fccb",
    )
    base.update(kw)
    return SessionSpec(**base)


@pytest.fixture(autouse=True)
def _isolated_output(tmp_path, monkeypatch):
    """Assemble into tmp_path instead of the station state dir.

    output_path() is state_dir/<GENERATED_DIR>/..., so without this the suite
    would write generated configs where a real station console looks for them
    -- and the console reopens the last generated config on /config. An
    artefact of one suite must not become input to another. pathlib resets on
    an absolute part, so an absolute first element replaces the state dir
    entirely.
    """
    monkeypatch.setattr(mod, "GENERATED_DIR", (str(tmp_path), "generated"))


def _template_cameras(preset_id="bimanual_v4"):
    """Every required camera of a preset's template, one device per block.

    assemble() rejects a form that leaves a declared camera unassigned, so a
    test that is not about camera assignment has to hand over a complete rig.
    Per block, not per name: every stream of one OAK block must agree on the
    board, and no two blocks may name the same device.

    A UVC block is identified by its V4L2 index rather than by a device name --
    there is no UVC registry -- so it gets the block's own number instead of a
    `fixture-device-N` string, which the assembler rightly refuses.
    """
    preset = sp.get_preset(preset_id)
    cfg = ruamel.YAML().load(template_path(preset, REPO).read_text())
    cameras = {}
    for index, source in enumerate(cfg.get("video_sources") or []):
        names = [s["name"] for s in source.get("streams") or [] if s.get("name")]
        if not names and source.get("name"):
            names = [source["name"]]
        uvc = source.get("type") == "uvc_camera"
        for name in names:
            if type_for_logical_name(name) is None:
                cameras[name] = str(index) if uvc else f"fixture-device-{index}"
    return cameras


def _devices(preset_id="bimanual_v4", **kw):
    """A complete rig for *preset_id*, with anything passed layered on top."""
    override = kw.pop("required_cameras", None)
    base = dict(
        grippers={"left": "UMI-V4-0001", "right": "UMI-V4-0002"},
        required_cameras=_template_cameras(preset_id),
        optional=(),
    )
    base.update(kw)
    if override is not None:
        base["required_cameras"] = {**base["required_cameras"], **override}
    return DeviceSpec(**base)


def _load(path):
    return ruamel.YAML().load(path.read_text())


def _source_names(cfg):
    """Every camera name the config records — what reference_camera is checked
    against.

    Source-level `name` and `streams[].name`, as
    RecorderManager._validate_reference_camera and
    reference_camera.video_source_names read them: a rig whose cameras are all
    oak_device streams would otherwise answer "records nothing" and could carry
    no valid reference at all. The shared function is called rather than
    reimplemented, so this cannot drift from the check it stands in for.
    """
    return video_source_names(cfg)


def _calibration_params(cfg):
    for check in cfg.get("pre_episode_checks") or []:
        if check.get("name") == "camera.relative_calibration":
            return check.get("params") or {}
    return {}


def _temp_repo(tmp_path, edit=None):
    """A repo root carrying its own copy of the templates.

    Several guards only fire for a config shape no shipped template has — a
    reference_camera anchored on a removable source, a stale optional source,
    a leftover language_instruction. They are reachable only against a copy,
    and the repository templates must stay untouched.

    *edit* rewrites the bimanual_v4 template, and must change it: an edit whose
    anchor text has drifted out of the template would otherwise hand the test
    the shipped file and let it pass without exercising anything.
    """
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copytree(TEMPLATE_DIR, root / "config" / "data_recording")
    if edit is not None:
        path = root / "config" / "data_recording" / "production" / "bimanual_v4.yaml"
        text = path.read_text()
        edited = edit(text)
        assert edited != text, "the edit no longer matches the template"
        path.write_text(edited)
    return root


def _insert_source(text, block):
    """Splice a video source in ahead of the template's first OAK block."""
    anchor = '  - type: oak_device\n    oak_device_name: "oak-left-wrist"'
    assert anchor in text
    return text.replace(anchor, block + anchor, 1)


_HEAD_IPHONE_SOURCE = """\
  - type: iphone_camera
    name: head_iphone_camera
    record_fps: 30
    stream_fps: 45
    frame_type: RGBDCameraFrame
    latency: 0.13
    params:
      iphone_device_name: "iphone15p-09"
      camera_resolution: [720, 960]

"""


# ---------------------------------------------------------------- meta block


def test_meta_records_preset_generation_and_uid(tmp_path):
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    meta = _load(out)["meta"]
    assert meta["setup_preset"] == "bimanual_v4"
    assert meta["operation_mode"] == "bimanual"
    assert meta["gripper_pipeline"] == "v4"
    assert meta["session_uid"] == "91e98c89fccb"


def test_meta_records_operator_and_task():
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    meta = _load(out)["meta"]
    assert meta["operator_name"] == "Test Operator"
    assert meta["task_name"] == "plane_assembly_v9"


@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_operator_is_recorded_as_unknown(blank):
    """Never the template's "Enter Name Here", and never an empty string the
    recorder would drop from the episode's attrs."""
    out = ConfigAssembler(REPO).assemble(_session(operator_name=blank), _devices())
    assert _load(out)["meta"]["operator_name"] == mod.UNKNOWN_OPERATOR == "Unknown"


def test_generation_comes_from_the_preset_not_the_device_name():
    out = ConfigAssembler(REPO).assemble(
        _session(preset_id="bimanual_v4"),
        _devices(grippers={"left": "RAW-SERIAL-A", "right": "RAW-SERIAL-B"}),
    )
    assert _load(out)["meta"]["gripper_pipeline"] == "v4"


def test_hardware_block_records_each_gripper():
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    hw = _load(out)["meta"]["hardware"]
    assert hw["left_umi_gripper"]["device"] == "UMI-V4-0001"
    assert hw["left_umi_gripper"]["generation"] == "v4"
    assert hw["right_umi_gripper"]["device"] == "UMI-V4-0002"


# Both V4 rigs carry the same OAK blocks: each arm's wrist camera is the only
# stream of an oak_device block of its own, video_sources[0] for the left arm
# and [1] for the right, so the right gripper keeps the fixture's second device.
@pytest.mark.parametrize(
    "preset_id, grippers, generation, override, left_camera, right_camera",
    [
        (
            preset_id,
            {"left": "UMI-V4-0001", "right": "UMI-V4-0002"},
            "v4",
            {"left_umi_camera": "oak-FFC-08"},
            "oak-FFC-08",
            "fixture-device-1",
        )
        for preset_id in ("bimanual_v4", "bimanual_v4_vive")
    ],
)
def test_hardware_block_records_the_camera_each_gripper_carried(
    preset_id, grippers, generation, override, left_camera, right_camera
):
    """Nothing camera-side reaches any episode attr -- record_manager.py
    writes gripper_serial and no camera identity at all -- so if a session ran
    on a board other than the linked one, this block is the only place it can
    be audited afterwards."""
    out = ConfigAssembler(REPO).assemble(
        _session(preset_id=preset_id),
        _devices(preset_id=preset_id, grippers=grippers, required_cameras=override),
    )
    hw = _load(out)["meta"]["hardware"]

    # Resolve camera associations per arm; a left-side override must not
    # change the right gripper's provenance.
    assert hw["left_umi_gripper"]["camera"] == left_camera
    assert hw["right_umi_gripper"]["camera"] == right_camera
    # The keys that were already there must survive.
    assert hw["left_umi_gripper"]["device"] == grippers["left"]
    assert hw["left_umi_gripper"]["generation"] == generation


def test_the_camera_key_is_omitted_when_an_arm_has_no_umi_camera():
    """A blank `camera:` reads as "recorded with no camera", which is a
    different and false claim.

    Driven through `_apply_meta` directly: no shipped preset declares a
    gripper arm without a `<arm>_umi_camera` stream, so `assemble` cannot
    reach this state and a test that went through it would be asserting on a
    rig that does not exist.
    """
    from ruamel.yaml.comments import CommentedMap

    # `_apply_meta` fills the template's own `meta` block; every shipped
    # template carries one.
    cfg = CommentedMap(meta=CommentedMap())
    ConfigAssembler._apply_meta(
        cfg,
        sp.get_preset("bimanual_v4"),
        _session(),
        DeviceSpec(grippers={"left": "UMI-V4-0001"}, required_cameras={}),
    )

    assert "camera" not in cfg["meta"]["hardware"]["left_umi_gripper"]
    assert cfg["meta"]["hardware"]["left_umi_gripper"]["device"] == "UMI-V4-0001"


# Assert the recorder-facing gripper_pipeline key and its value; an
# incorrect key can silently select the parser's default.


@pytest.mark.parametrize("preset_id", ["bimanual_v4", "bimanual_v4_vive"])
def test_the_recorder_reads_back_the_pipeline_the_preset_selected(preset_id):
    out = ConfigAssembler(REPO).assemble(
        _session(preset_id=preset_id), _devices(preset_id)
    )
    written = _load(out)["meta"]["gripper_pipeline"]
    assert written == "v4"
    assert GripperPipeline.is_known(written)
    assert GripperPipeline.parse(written) is GripperPipeline.V4


def test_meta_carries_no_second_spelling_of_the_pipeline():
    """One key, or the two can disagree.

    A second key such as `gripper_generation`, read by nothing, is worse than
    none: the next reader picks the one that is wrong and there is no test to
    catch it.
    """
    meta = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))["meta"]
    assert "gripper_generation" not in meta


def test_a_bad_session_uid_is_rejected():
    for bad in ("", "nope", "91E98C89FCCB", "91e98c89fcc"):
        with pytest.raises(ValueError):
            ConfigAssembler(REPO).assemble(_session(session_uid=bad), _devices())


def test_use_head_mount_follows_the_head_camera_section(tmp_path):
    """Derived from the head section, never copied from the template. A
    template that ships `use_head_mount: true` would otherwise have a rig with
    no head device claim a head mount it does not have, and the QR gate then
    schedules a camera that is not in the file -- so the copy here ships `true`,
    which is the direction a copied-through value would get wrong."""
    repo = _temp_repo(
        tmp_path,
        lambda text: text.replace(
            "  use_head_mount: false", "  use_head_mount: true", 1
        ),
    )
    assembler = ConfigAssembler(repo)
    bare = _load(assembler.assemble(_session(), _devices()))
    assert bare["meta"]["use_head_mount"] is False

    with_head = _load(
        assembler.assemble(
            _session(),
            _devices(
                optional=(
                    OptionalDeviceAssignment(
                        "head_iphone", "head_iphone_camera", "iphone16p-ry"
                    ),
                ),
            ),
        )
    )
    assert with_head["meta"]["use_head_mount"] is True

    # A bottom camera is not a head mount.
    with_bottom = _load(
        assembler.assemble(
            _session(),
            _devices(
                optional=(
                    OptionalDeviceAssignment(
                        "bottom_oak", "bottom_oak_camera", "oak-W97-52", "table"
                    ),
                ),
            ),
        )
    )
    assert with_bottom["meta"]["use_head_mount"] is False


def test_a_stale_language_instruction_is_dropped(tmp_path):
    """Superseded by annotation.human.task_description in the dataset; a copy
    left in meta only misleads whoever reads the recording config."""
    repo = _temp_repo(
        tmp_path,
        lambda text: text.replace(
            "  use_head_mount: false",
            '  language_instruction: "pick up the cup"\n  use_head_mount: false',
            1,
        ),
    )
    cfg = _load(ConfigAssembler(repo).assemble(_session(), _devices()))
    assert "language_instruction" not in cfg["meta"]


def test_template_only_keys_are_preserved():
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    cfg = _load(out)
    # A from-scratch writer would drop both of these.
    assert "pre_episode_checks" in cfg
    assert "reference_camera" in cfg


def test_the_alignment_grid_rides_through_from_the_template(tmp_path):
    """alignment.uniform_fps is what the background aligner reads: copied
    from the template where it sets it, and absent where it does not (the
    aligner then uses its 30 fps default). The assembler invents neither."""
    v4 = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))
    assert v4["alignment"]["uniform_fps"] == 30
    repo = _temp_repo(
        tmp_path,
        lambda text: text.replace(
            "  uniform_fps: 30      # aligned rows are an even 1/30 s apart\n",
            "",
            1,
        ),
    )
    unset = _load(ConfigAssembler(repo).assemble(_session(), _devices()))
    assert "uniform_fps" not in unset["alignment"]


# ------------------------------------------------------------ meta.HW_version


def _registered(monkeypatch, hw_version, task_name="plane_assembly_v9"):
    """Answer the task registry with one task carrying *hw_version*.

    Patched at the store so the real registry code still runs. Without it the
    answer would depend on whatever registry the run resolves (tests/conftest.py
    points the registry file at an absent path), not on this test.
    """
    monkeypatch.setattr(
        task_registry_store,
        "get_all_tasks",
        lambda: {
            task_name: {
                "name": task_name,
                "short_description": "Assemble the plane",
                "mode": "bimanual",
                "HW_version": hw_version,
                "training_group": task_name,
            }
        },
    )


def test_hw_version_follows_the_rig_when_the_registry_disagrees(monkeypatch):
    """The task registry's `HW_version` is a per-task label, and many tasks
    still carry the v1 they were registered under. Taking it alone would stamp
    a V4 bench's session v1, contradicting `gripper_pipeline` and
    `hardware.*.generation` a few lines below it in the same file."""
    _registered(monkeypatch, "v1")
    meta = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))["meta"]
    assert meta["HW_version"] == "v4"


def test_hw_version_keeps_the_registry_minor_when_the_major_agrees(monkeypatch):
    """`v4.1` says something no preset can: a preset knows only `v4`. Agreeing
    on the major, the registry's fuller label is the better of the two."""
    _registered(monkeypatch, "v4.1")
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    assert _load(out)["meta"]["HW_version"] == "v4.1"


def test_hw_version_falls_back_to_the_rig_for_an_unregistered_task(monkeypatch):
    """An unknown task yields "" from the registry, so the rig's version is used."""
    monkeypatch.setattr(task_registry_store, "get_all_tasks", dict)
    meta = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))["meta"]
    assert meta["HW_version"] == "v4"


def test_an_explicitly_supplied_hw_version_still_wins(monkeypatch):
    """The caller-resolved value is an override, and stays one."""
    _registered(monkeypatch, "v1")
    out = ConfigAssembler(REPO).assemble(_session(hw_version="v4.2"), _devices())
    assert _load(out)["meta"]["HW_version"] == "v4.2"


# -------------------------------------------------------- meta.tracking_source


def test_tracking_source_names_the_iphone_behind_each_arms_pose():
    """Every slot is written, the head too: with no head camera it says
    `notracker`, a claim about the rig rather than a missing key, so a consumer
    indexes the block instead of testing for it."""
    meta = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))["meta"]
    assert meta["tracking_source"] == {
        "head": "notracker",
        "left": "iphone",
        "right": "iphone",
    }


def test_a_head_iphone_is_a_head_tracking_source():
    """Every iphone_camera records an ARKit pose (iphone_camera.py), so a head
    phone carries head pose exactly as an arm phone carries arm pose."""
    out = ConfigAssembler(REPO).assemble(
        _session(),
        _devices(
            optional=(
                OptionalDeviceAssignment(
                    "head_iphone", "head_iphone_camera", "iphone16p-ry"
                ),
            )
        ),
    )
    assert _load(out)["meta"]["tracking_source"]["head"] == "iphone"


def test_a_head_oak_carries_no_pose():
    out = ConfigAssembler(REPO).assemble(
        _session(),
        _devices(
            optional=(
                OptionalDeviceAssignment("head_oak", "head_oak_camera", "oak-W97-51"),
            )
        ),
    )
    assert _load(out)["meta"]["tracking_source"]["head"] == "notracker"


# -------------------------------------------------------------- recording


def test_recording_records_the_session_dir():
    out = ConfigAssembler(REPO).assemble(
        _session(data_dir="output/raw/plane_assembly_v9/s1"),
        _devices(),
    )
    rec = _load(out)["recording"]
    assert rec["data_dir"] == "output/raw/plane_assembly_v9/s1"
    assert "auto_uploader" not in rec


# ---------------------------------------------------------- optional devices


def test_head_cameras_are_emitted_with_placement():
    # Read back off the written file, because what matters is what a downstream
    # consumer finds there, not what the assembler intended.
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "oak-W97-51", "head"
            ),
            OptionalDeviceAssignment(
                "head_iphone", "head_iphone_camera", "iphone16p-ry", "neck"
            ),
        )
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    by_name = {s["name"]: s for s in cfg["video_sources"] if "name" in s}
    assert by_name["head_oak_camera"]["placement"] == "head"
    assert by_name["head_iphone_camera"]["placement"] == "neck"


def test_a_head_camera_may_not_take_a_placement_its_section_does_not_offer():
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "oak-W97-51", "left_front"
            ),
        )
    )
    with pytest.raises(ValueError, match="head"):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_a_bottom_camera_may_not_take_a_head_only_placement():
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "bottom_oak", "bottom_oak_camera", "oak-W97-52", "neck"
            ),
        )
    )
    with pytest.raises(ValueError, match="bottom"):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_a_spoken_placement_is_normalised_rather_than_written_through():
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "oak-W97-51", " Neck "
            ),
        )
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    head = next(s for s in cfg["video_sources"] if s.get("name") == "head_oak_camera")
    assert head["placement"] == "neck"


def test_a_placement_written_before_the_split_still_assembles():
    # Persisted head-placement aliases must still load even when new forms
    # offer a narrower choice set.
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "oak-W97-51", "overhead"
            ),
        )
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    head = next(s for s in cfg["video_sources"] if s.get("name") == "head_oak_camera")
    assert head["placement"] == "overhead"


def test_a_placement_round_trips_through_the_written_file():
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "oak-W97-51", "static"
            ),
            OptionalDeviceAssignment(
                "bottom_oak", "bottom_oak_camera", "oak-W97-52", "table"
            ),
        )
    )
    out = ConfigAssembler(REPO).assemble(_session(), devices)

    from station_ui import config_form
    from station_ui.session import SessionDraft

    draft = SessionDraft()
    config_form.hydrate(draft, _load(out))
    by_name = {a.logical_name: a for a in draft.optional}
    assert by_name["head_oak_camera"].placement == "static"
    assert by_name["bottom_oak_camera"].placement == "table"


def test_an_explicit_reference_override_round_trips_through_the_written_file():
    # The operator anchored on a tracking camera rather than the automatic pick,
    # so reopening the written config must restore that choice as an override
    # rather than mistake it for the automatic pick.
    devices = _devices(
        reference_override="right_tracking_camera",
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "oak-W97-51", "head"
            ),
        ),
    )
    out = ConfigAssembler(REPO).assemble(_session(), devices)

    from station_ui import config_form
    from station_ui.session import SessionDraft

    draft = SessionDraft()
    config_form.hydrate(draft, _load(out))
    assert draft.reference_override == "right_tracking_camera"


def test_an_automatic_reference_reopens_as_automatic_not_a_sticky_override():
    # The operator accepted the auto-pick. Reopening must not freeze it as an
    # override, or removing the head camera later would no longer move the
    # anchor.
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "oak-W97-51", "head"
            ),
        )
    )
    out = ConfigAssembler(REPO).assemble(_session(), devices)

    from station_ui import config_form
    from station_ui.session import SessionDraft

    draft = SessionDraft()
    config_form.hydrate(draft, _load(out))
    assert draft.reference_override == ""


def test_head_cameras_are_emitted_with_fixed_names():
    devices = _devices(
        optional=(
            OptionalDeviceAssignment("head_oak", "head_oak_camera", "oak-W97-51"),
            OptionalDeviceAssignment(
                "head_iphone", "head_iphone_camera", "iphone16p-ry"
            ),
        )
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    names = {s.get("name") for s in cfg["video_sources"]}
    assert {"head_oak_camera", "head_iphone_camera"} <= names


def test_head_camera_blocks_carry_their_device_identifier():
    devices = _devices(
        optional=(
            OptionalDeviceAssignment("head_oak", "head_oak_camera", "oak-W97-51"),
            OptionalDeviceAssignment(
                "head_iphone", "head_iphone_camera", "iphone16p-ry"
            ),
        )
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    by_name = {s["name"]: s for s in cfg["video_sources"] if "name" in s}
    assert by_name["head_oak_camera"]["type"] == "oak_camera"
    assert by_name["head_oak_camera"]["params"]["oak_device_name"] == "oak-W97-51"
    assert by_name["head_iphone_camera"]["type"] == "iphone_camera"
    assert (
        by_name["head_iphone_camera"]["params"]["iphone_device_name"] == "iphone16p-ry"
    )


def test_a_raw_hardware_id_is_stored_under_the_id_key():
    # Unregistered hardware has no registry name, so the GUI hands over the raw
    # id. Writing it under *_device_name makes the name->id lookup guess.
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "14442C1000000000AF"
            ),
        )
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    head = next(s for s in cfg["video_sources"] if s.get("name") == "head_oak_camera")
    assert head["params"]["oak_device_id"] == "14442C1000000000AF"
    assert "oak_device_name" not in head["params"]


def test_no_optional_devices_means_no_optional_sources():
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))
    names = {s.get("name") for s in cfg["video_sources"]}
    assert not names & {"head_oak_camera", "head_iphone_camera", "bottom_oak_camera"}


def test_an_unknown_placement_is_rejected():
    devices = _devices(
        optional=(
            OptionalDeviceAssignment(
                "head_oak", "head_oak_camera", "oak-W97-51", "upside_down"
            ),
        )
    )
    with pytest.raises(ValueError):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_two_optional_devices_may_not_share_a_name():
    # Each optional camera is written under its descriptor's one fixed name, so
    # a second row of the same type collides on it. Distinct devices, so the
    # collision is the name and not the device.
    devices = _devices(
        optional=(
            OptionalDeviceAssignment("head_oak", "head_oak_camera", "oak-W97-51"),
            OptionalDeviceAssignment("head_oak", "head_oak_camera", "oak-W97-52"),
        )
    )
    with pytest.raises(ValueError):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_no_optional_camera_leaves_the_calibration_check_untouched():
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))
    params = _calibration_params(cfg)
    assert set(params.get("cameras") or []) == {
        "right_tracking_camera",
        "left_tracking_camera",
    }


# ------------------------------------------------ the world-frame anchor


def test_a_bare_v4_rig_anchors_on_the_wrist_tracking_camera():
    """No fixed camera at all, so the anchor falls through the head to the
    tracking cameras, right before left -- the same name the template ships,
    arrived at by the rule rather than copied. Never a trigger camera: the USB
    module reports no intrinsics, so it can never solve the board."""
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))
    assert cfg["reference_camera"] == "right_tracking_camera"
    assert _calibration_params(cfg)["reference_camera"] == "right_tracking_camera"


def test_a_bare_vive_rig_anchors_on_the_right_wrist_camera():
    """No head camera and no phone: the wrist OAKs are the only cameras that
    can solve the board, right before left. The USB triggers record slower and
    still do not take it."""
    cfg = _load(
        ConfigAssembler(REPO).assemble(
            _session(preset_id="bimanual_v4_vive"), _devices("bimanual_v4_vive")
        )
    )
    assert cfg["reference_camera"] == "right_umi_camera"
    params = _calibration_params(cfg)
    assert params["reference_camera"] == "right_umi_camera"
    assert not {"left_trigger_camera", "right_trigger_camera"} & set(
        params.get("cameras") or []
    )


# ------------------------------------- the world-frame anchor is the slowest cam


def test_the_rate_is_read_off_record_fps_not_the_sensor_fps():
    """Unit-level, so the two are separated by name and not just by outcome.

    The slow camera is driven at the fastest number in the file (`fps: 60`) and
    recorded at the slowest; reading the sensor rate instead would invert the
    answer. It is a wrist camera on purpose: it loses to the tracking camera on
    every tie-break, so only the rate can make it the anchor."""
    cfg = {
        "recording": {"record_fps": 30},
        "video_sources": [
            {
                "type": "iphone_camera",
                "name": "right_tracking_camera",
                "record_fps": 30,
            },
            {
                "type": "oak_device",
                "fps": 60,
                "record_fps": 25,
                "streams": [{"name": "left_umi_camera"}],
            },
            {
                "type": "oak_device",
                "fps": 45,
                "streams": [{"name": "right_umi_camera"}],
            },
        ],
    }
    assert ConfigAssembler._record_rates(cfg) == {
        "right_tracking_camera": 30,
        "left_umi_camera": 25,
        # Inherited from recording.record_fps -- the block's `fps: 45` is the
        # rate the board is driven at, not the rate frames are written.
        "right_umi_camera": 30,
    }
    assert ConfigAssembler._auto_reference(cfg) == "left_umi_camera"


def test_a_usb_trigger_camera_never_takes_the_world_frame():
    """The slowest camera of a V4 rig is its USB trigger, recorded at 25, and it
    must not anchor all the same: the module reports no intrinsics, so it can
    never solve the board the relative calibration measures every camera
    against. The rate rule chooses among the cameras that can."""
    cfg = {
        "recording": {"record_fps": 30},
        "video_sources": [
            {
                "type": "uvc_camera",
                "name": "right_trigger_camera",
                "record_fps": 25,
                "params": {"fps": 30},
            },
            {
                "type": "iphone_camera",
                "name": "left_tracking_camera",
                "record_fps": 30,
            },
        ],
    }
    assert ConfigAssembler._auto_reference(cfg) == "left_tracking_camera"


def test_the_tie_break_runs_head_tracking_wrist_trigger():
    """All at one rate, so nothing but rank decides. Asserted as a whole order
    rather than pairwise: the chain is the specification."""
    rank = ConfigAssembler._reference_rank
    order = [
        "head_oak_camera",
        "right_tracking_camera",
        "left_tracking_camera",
        "right_umi_camera",
        "left_umi_camera",
        "right_trigger_camera",
        "left_trigger_camera",
    ]
    assert sorted(order, key=rank) == order
    # And reversing the input cannot change the answer -- the key is total.
    assert sorted(reversed(order), key=rank) == order


def test_a_camera_with_no_rate_anywhere_sorts_last_not_first():
    """Unknown is not slow. Treating a missing record_fps as zero would hand the
    world frame to whichever camera is least described."""
    cfg = {
        "video_sources": [
            {"type": "oak_device", "streams": [{"name": "left_umi_camera"}]},
            {
                "type": "iphone_camera",
                "name": "right_tracking_camera",
                "record_fps": 30,
            },
        ],
    }
    assert ConfigAssembler._auto_reference(cfg) == "right_tracking_camera"


# ------------------------------------------- head camera as world-frame anchor


def _head_oak(device="oak-W97-51", placement="head"):
    return OptionalDeviceAssignment("head_oak", "head_oak_camera", device, placement)


def test_a_head_oak_becomes_the_world_frame_reference():
    """It is the most rigidly fixed camera the rig can carry -- bolted on, aimed
    down the workspace, and not nudged between episodes. Written to both the
    top-level reference_camera and the calibration check's own, the two places
    RecorderManager reads."""
    cfg = _load(
        ConfigAssembler(REPO).assemble(_session(), _devices(optional=(_head_oak(),)))
    )
    assert cfg["reference_camera"] == "head_oak_camera"
    assert _calibration_params(cfg)["reference_camera"] == "head_oak_camera"


def test_the_head_reference_joins_the_calibration_rig():
    """A camera the world frame is measured from that is not itself calibrated
    is a frame nothing is actually relative to."""
    cfg = _load(
        ConfigAssembler(REPO).assemble(_session(), _devices(optional=(_head_oak(),)))
    )
    assert "head_oak_camera" in set(_calibration_params(cfg)["cameras"])


def _bottom_oak(device="oak-W97-77", placement="table"):
    return OptionalDeviceAssignment(
        "bottom_oak", "bottom_oak_camera", device, placement
    )


def test_a_bottom_oak_is_emitted_like_the_head_oak():
    """Same `oak_camera` block as the head, under its own fixed name, carrying
    the operator's device and where it stands. Nothing downstream learns a new
    source type -- and it is not a head mount, so the flag the QR gate and the
    worker read stays false."""
    cfg = _load(
        ConfigAssembler(REPO).assemble(_session(), _devices(optional=(_bottom_oak(),)))
    )
    block = next(
        s for s in cfg["video_sources"] if s.get("name") == "bottom_oak_camera"
    )
    assert block["type"] == "oak_camera"
    assert block["params"]["oak_device_name"] == "oak-W97-77"
    assert block["placement"] == "table"
    assert cfg["meta"]["use_head_mount"] is False


def test_a_bottom_oak_never_anchors_the_world_frame():
    """It looks up from the table, so there is no promise it sees the board the
    other cameras share. It records; the frame is anchored as if it were absent,
    and it stays out of the relative-calibration rig."""
    alone = _load(
        ConfigAssembler(REPO).assemble(_session(), _devices(optional=(_bottom_oak(),)))
    )
    assert alone["reference_camera"] == "right_tracking_camera"
    assert "bottom_oak_camera" not in set(_calibration_params(alone)["cameras"])

    with_head = _load(
        ConfigAssembler(REPO).assemble(
            _session(), _devices(optional=(_bottom_oak(), _head_oak()))
        )
    )
    assert with_head["reference_camera"] == "head_oak_camera"
    assert {"head_oak_camera", "bottom_oak_camera"} <= _source_names(with_head)


def test_a_head_oak_and_a_bottom_oak_may_not_share_one_board():
    """Two fixed OAKs on one bench, and both pickers list the same pool."""
    with pytest.raises(ValueError, match="oak-W97-51"):
        ConfigAssembler(REPO).assemble(
            _session(),
            _devices(optional=(_head_oak("oak-W97-51"), _bottom_oak("oak-W97-51"))),
        )


def test_the_head_oak_anchors_the_vive_rig_too():
    """The rig that most needs it: with no phones, every other camera it records
    moves with an arm. Its template names the head OAK as its anchor."""
    devices = _devices("bimanual_v4_vive", optional=(_head_oak(),))
    cfg = _load(
        ConfigAssembler(REPO).assemble(_session(preset_id="bimanual_v4_vive"), devices)
    )
    assert cfg["reference_camera"] == "head_oak_camera"


def test_the_picker_offers_the_head_camera_and_defaults_to_it():
    """The form must agree with the file. `reference_choices` renders on a
    half-filled draft, so it computes the same pick a different way -- off the
    assignments rather than off assembled sources -- and the two drifting is
    what would show an operator "Automatic — Right tracking camera" above a
    config anchored on the head."""
    devices = _devices(optional=(_head_oak(),))
    choices = ConfigAssembler(REPO).reference_choices("bimanual_v4", devices)

    assert choices.auto == "head_oak_camera"
    assert choices.candidates[0].name == "head_oak_camera"
    assert "right_tracking_camera" in {c.name for c in choices.candidates}


def test_the_picker_ignores_a_head_row_with_no_device_chosen():
    """A row added and never filled in is not a camera the session records, so
    it must not be offered as an anchor or shown as the automatic pick.

    Asserted on the picker rather than on `assemble`, which refuses such a draft
    outright -- but the picker renders on exactly the half-filled form that
    refusal describes, so it is the one that has to cope."""
    devices = _devices(optional=(_head_oak(device=""),))
    choices = ConfigAssembler(REPO).reference_choices("bimanual_v4", devices)

    assert choices.auto == "right_tracking_camera"
    assert "head_oak_camera" not in {c.name for c in choices.candidates}


# ------------------------------------------ operator override of the reference


def test_reference_override_is_honoured_over_the_auto_pick():
    # The head OAK is the auto-pick. The operator names a wrist camera instead,
    # and it is honoured -- in both the top-level reference_camera and the
    # calibration check's -- and it joins the rig.
    devices = _devices(
        reference_override="right_umi_camera",
        optional=(_head_oak(),),
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    assert cfg["reference_camera"] == "right_umi_camera"
    assert _calibration_params(cfg)["reference_camera"] == "right_umi_camera"
    assert "right_umi_camera" in set(_calibration_params(cfg)["cameras"])


def test_an_empty_override_reproduces_the_auto_pick_exactly():
    optional = (_head_oak(),)
    auto = _load(
        ConfigAssembler(REPO).assemble(_session(), _devices(optional=optional))
    )
    empty = _load(
        ConfigAssembler(REPO).assemble(
            _session(), _devices(reference_override="", optional=optional)
        )
    )
    assert empty["reference_camera"] == auto["reference_camera"] == "head_oak_camera"
    assert (
        _calibration_params(empty)["reference_camera"]
        == _calibration_params(auto)["reference_camera"]
    )


def test_an_override_that_names_an_absent_camera_is_refused():
    # A reference naming a source the rig does not record aborts recording at
    # construction; caught here, at assemble, with a message that names the
    # camera the operator has to fix rather than 500ing later.
    devices = _devices(reference_override="nonexistent_camera")
    with pytest.raises(ValueError, match="nonexistent_camera"):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_an_override_to_a_tracking_camera_works_with_no_head_camera():
    # No head camera: the auto-pick is right_tracking_camera. The operator names
    # left_tracking_camera and it takes over.
    devices = _devices(reference_override="left_tracking_camera")
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    assert cfg["reference_camera"] == "left_tracking_camera"
    assert _calibration_params(cfg)["reference_camera"] == "left_tracking_camera"


def test_a_logical_name_that_does_not_match_its_descriptor_is_rejected():
    devices = _devices(
        optional=(
            OptionalDeviceAssignment("head_oak", "head_tracking_camera", "oak-W97-51"),
        )
    )
    with pytest.raises(ValueError):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_a_stale_optional_source_in_the_loaded_config_is_dropped(tmp_path):
    """Anything descriptor-shaped is removed before the operator's choices are
    appended, so a config that already carries one cannot end up with the
    camera the operator just deleted still recording."""
    repo = _temp_repo(tmp_path, lambda text: _insert_source(text, _HEAD_IPHONE_SOURCE))
    cfg = _load(ConfigAssembler(repo).assemble(_session(), _devices()))
    assert "head_iphone_camera" not in _source_names(cfg)


# ------------------------------------------------- one device, one source block


def test_a_required_and_an_optional_camera_may_not_share_one_device():
    """Two IphoneCameras session.connect() to one Record3D device and one of
    them silently records nothing useful — fail-silent, so the assembler has to
    fail loud instead."""
    devices = _devices(
        required_cameras={"right_tracking_camera": "iphone16p-ry"},
        optional=(
            OptionalDeviceAssignment(
                "head_iphone", "head_iphone_camera", "iphone16p-ry"
            ),
        ),
    )
    with pytest.raises(ValueError, match="iphone16p-ry"):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_a_device_shared_across_the_two_halves_writes_nothing():
    assembler = ConfigAssembler(REPO)
    good = assembler.assemble(_session(), _devices()).read_bytes()
    with pytest.raises(ValueError):
        assembler.assemble(
            _session(),
            _devices(
                required_cameras={"right_tracking_camera": "iphone16p-ry"},
                optional=(
                    OptionalDeviceAssignment(
                        "head_iphone", "head_iphone_camera", "iphone16p-ry"
                    ),
                ),
            ),
        )
    assert assembler.output_path(_session()).read_bytes() == good


def test_distinct_devices_across_the_two_halves_are_accepted():
    devices = _devices(
        required_cameras={"right_tracking_camera": "iphone16p-ry"},
        optional=(
            OptionalDeviceAssignment(
                "head_iphone", "head_iphone_camera", "iphone16p-zz"
            ),
        ),
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    by_name = {s["name"]: s for s in cfg["video_sources"] if "name" in s}
    assert (
        by_name["right_tracking_camera"]["params"]["iphone_device_name"]
        == "iphone16p-ry"
    )
    assert (
        by_name["head_iphone_camera"]["params"]["iphone_device_name"] == "iphone16p-zz"
    )


# ------------------------------------------------------ device-scan warnings


def test_a_resolution_the_scan_could_not_confirm_is_reported():
    """resolution_for returns why it fell back so the caller can surface it. An
    unscanned 4:3 IMX378 written as [640, 400] loses 17% of its FOV, and the
    explanation is the only sign that happened."""
    assembler = ConfigAssembler(REPO)
    assembler.assemble(
        _session(),
        _devices(
            optional=(
                OptionalDeviceAssignment("head_oak", "head_oak_camera", "oak-W97-51"),
            )
        ),
    )
    assert any("oak-W97-51" in w for w in assembler.last_resolution_warnings)


def test_the_head_oak_resolution_follows_its_cam_a_sensor_module():
    """`oak_camera` streams CAM_A, so the head OAK's camera_resolution is that
    socket's sensor aspect: a 4:3 IMX378 is 640x480, a 16:10 AR0234 is 640x400.
    This pins the success path, not just the fallback warning, so a break cannot
    write a cropped [640, 400] in silence."""
    from station_ui.core.device_scan import DeviceScanner

    ser = "14442C1000000000AF"

    def head_res(module):
        scanner = DeviceScanner.from_records(
            [
                {
                    "kind": "oak",
                    "device_id": ser,
                    "registry_name": None,
                    "state": "available",
                    "metadata": {"sockets": {"CAM_A": module}},
                }
            ]
        )
        cfg = _load(
            ConfigAssembler(REPO, scanner=scanner).assemble(
                _session(),
                _devices(
                    optional=(
                        OptionalDeviceAssignment("head_oak", "head_oak_camera", ser),
                    )
                ),
            )
        )
        head = next(
            s for s in cfg["video_sources"] if s.get("name") == "head_oak_camera"
        )
        return head["params"]["camera_resolution"]

    assert head_res("IMX378") == [640, 480]  # 4:3
    assert head_res("AR0234") == [640, 400]  # 16:10


def test_a_required_camera_reports_its_resolution_warning_once():
    assembler = ConfigAssembler(REPO)
    assembler.assemble(
        _session(),
        _devices(required_cameras={"left_umi_camera": "oak-FFC-22"}),
    )
    warnings = assembler.last_resolution_warnings
    # One board, one warning. Counted per board rather than globally: a
    # complete rig has a second OAK block, which warns about its own device
    # and says nothing about de-duplication.
    assert len([w for w in warnings if "oak-FFC-22" in w]) == 1, warnings


def test_warnings_describe_only_the_most_recent_assembly():
    assembler = ConfigAssembler(REPO)
    assembler.assemble(
        _session(),
        _devices(required_cameras={"left_umi_camera": "oak-FFC-22"}),
    )
    assert any("oak-FFC-22" in w for w in assembler.last_resolution_warnings)
    assembler.assemble(_session(), _devices())
    assert not any("oak-FFC-22" in w for w in assembler.last_resolution_warnings)


# ------------------------------------------------------- required hardware


def test_required_cameras_are_written_onto_the_matching_sources():
    devices = _devices(
        required_cameras={
            "left_umi_camera": "oak-FFC-22",
            "right_tracking_camera": "iphone15p-99",
        }
    )
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), devices))
    ffc = next(
        s
        for s in cfg["video_sources"]
        if s.get("type") == "oak_device"
        and any(st["name"] == "left_umi_camera" for st in s["streams"])
    )
    assert ffc["oak_device_name"] == "oak-FFC-22"
    phone = next(
        s for s in cfg["video_sources"] if s.get("name") == "right_tracking_camera"
    )
    assert phone["params"]["iphone_device_name"] == "iphone15p-99"


def test_a_required_camera_the_template_does_not_declare_is_rejected():
    # Silently dropping it would record from the template's placeholder device.
    devices = _devices(required_cameras={"nose_camera": "oak-FFC-22"})
    with pytest.raises(ValueError):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_one_physical_camera_may_not_back_two_source_blocks():
    # The OAK device cache hands the second block the same handle, and every
    # other driver simply fails to open the device twice.
    devices = _devices(
        required_cameras={
            "left_umi_camera": "oak-FFC-22",
            "right_umi_camera": "oak-FFC-22",
        }
    )
    with pytest.raises(ValueError, match="oak-FFC-22"):
        ConfigAssembler(REPO).assemble(_session(), devices)


def test_a_rejected_camera_assignment_repoints_nothing():
    assembler = ConfigAssembler(REPO)
    good = assembler.assemble(_session(), _devices()).read_bytes()
    with pytest.raises(ValueError):
        assembler.assemble(
            _session(),
            _devices(
                required_cameras={
                    "left_umi_camera": "oak-FFC-22",
                    "nose_camera": "oak-FFC-33",
                }
            ),
        )
    assert assembler.output_path(_session()).read_bytes() == good


def test_grippers_are_written_onto_the_existing_numeric_sources():
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    numeric = {n["name"]: n for n in _load(out)["numeric_sources"]}
    assert numeric["left_umi_gripper"]["uart_device_name"] == "UMI-V4-0001"
    assert numeric["right_umi_gripper"]["uart_device_name"] == "UMI-V4-0002"


def test_a_gripper_arm_the_template_does_not_declare_is_rejected():
    # The template declares a left and a right gripper; a silently dropped third
    # arm would leave meta.hardware claiming a gripper that never records.
    with pytest.raises(ValueError):
        ConfigAssembler(REPO).assemble(
            _session(),
            _devices(
                grippers={
                    "left": "UMI-V4-0001",
                    "right": "UMI-V4-0002",
                    "head": "UMI-V4-0003",
                }
            ),
        )


# -------------------------------------------------- templates and topology


def test_output_goes_under_generated_and_never_touches_the_template():
    assembler = ConfigAssembler(REPO)
    session = _session()
    template = TEMPLATE_DIR / "production" / "bimanual_v4.yaml"
    before = template.read_bytes()
    out = assembler.assemble(session, _devices())
    assert "generated" in out.parts
    assert template.read_bytes() == before


def test_every_repository_template_survives_assembly_unmodified():
    # Recursive: the preset templates live in `production/`, below TEMPLATE_DIR.
    before = {
        p.relative_to(TEMPLATE_DIR): p.read_bytes()
        for p in sorted(TEMPLATE_DIR.rglob("*.yaml"))
    }
    assert any(path.parts[0] == "production" for path in before), before.keys()
    assembler = ConfigAssembler(REPO)
    # The Vive rig's two numeric sources are the only ones the console writes
    # `params` into, so it is the shape most likely to mutate a template in the
    # checkout.
    for preset_id in ("bimanual_v4", "bimanual_v4_vive"):
        assembler.assemble(_session(preset_id=preset_id), _devices(preset_id))
    after = {
        p.relative_to(TEMPLATE_DIR): p.read_bytes()
        for p in sorted(TEMPLATE_DIR.rglob("*.yaml"))
    }
    assert after == before


@pytest.mark.parametrize("preset_id", ["bimanual_v4", "bimanual_v4_vive"])
def test_both_v4_rigs_capture_oak_at_30_and_record_at_30(preset_id):
    """The two V4 rigs share their OAK blocks and their rate.

    Three numbers, and they are three different things:

      * `fps` is what the board captures at. 30, with the AE ceiling kept at
        the 1/45 s the 45 fps capture had.
      * `stream_fps` is how often the stream thread polls. 45 -- above the
        capture rate, so the read blocks until a frame arrives.
      * `record_fps` is what reaches the mp4. 30.

    Capture == record keeps one frame per record tick; capturing faster than
    recording leaves recorded frames alternating 22 / 44 ms apart.

    Pinned on every save rather than only inherited from the template
    (`_pin_oak_rates`): a config generated by an older version is reopened,
    not regenerated, so without the pin an existing station would keep its old
    capture rate until somebody deleted the file by hand.

    latency: 0.08 is a measured property of the pipeline, not a function of the
    frame rate -- which is why it is asserted as a number of its own rather than
    derived from the rates.
    """
    cfg = _load(
        ConfigAssembler(REPO).assemble(
            _session(preset_id=preset_id), _devices(preset_id)
        )
    )
    oak_blocks = [s for s in cfg["video_sources"] if s.get("type") == "oak_device"]
    assert oak_blocks, "expected oak_device blocks in the template"
    assert all(b["fps"] == 30 for b in oak_blocks), [b["fps"] for b in oak_blocks]
    assert all(b["ae_max_exposure_us"] == 22222 for b in oak_blocks)
    # An oak_device states no rates of its own, in the template or here:
    # it inherits the global, which the next test pins.
    assert not any("record_fps" in b for b in oak_blocks)
    assert all(b["latency"] == 0.08 for b in oak_blocks), [
        b["latency"] for b in oak_blocks
    ]


def test_the_v4_rates_reach_every_source_and_the_global_default():
    """Every source records and streams at the rate its type is meant to.

    Nothing records faster than 30. The USB trigger cameras record at 25 and
    stream at 30, what the module delivers with its default auto-exposure; the
    iPhones are polled at 90 -- above Record3D's 60 Hz; everything else streams
    at the 45 default.

    The global is only a default: most sources in the template state their own
    rates, so lowering `recording.record_fps` alone would have moved nothing.
    This walks what was actually written rather than trusting that.
    """
    cfg = _load(ConfigAssembler(REPO).assemble(_session(), _devices()))

    assert cfg["recording"]["record_fps"] == 30
    assert cfg["recording"]["stream_fps"] == 45

    expected = {"uvc_camera": (25, 30), "iphone_camera": (30, 90)}
    for source in list(cfg["video_sources"]) + list(cfg.get("numeric_sources") or []):
        name = source.get("name") or source.get("oak_device_name")
        record, stream = expected.get(source.get("type"), (30, 45))
        assert source.get("record_fps", 30) == record, (name, source.get("record_fps"))
        assert source.get("stream_fps", 45) == stream, (name, source.get("stream_fps"))


# ------------------------------------------- every rig records its reference


def test_every_preset_leaves_a_resolvable_reference_camera():
    assembler = ConfigAssembler(REPO)
    for preset_id in sp.preset_ids():
        for optional in (
            (),
            (
                OptionalDeviceAssignment(
                    "head_iphone", "head_iphone_camera", "iphone16p-ry"
                ),
            ),
        ):
            cfg = _load(
                assembler.assemble(
                    _session(preset_id=preset_id),
                    _devices(preset_id, optional=optional),
                )
            )
            names = _source_names(cfg)
            label = (preset_id, bool(optional))
            assert cfg["reference_camera"] in names, label
            cams = _calibration_params(cfg).get("cameras") or []
            assert not set(cams) - names, label


def _anchored_on_a_removable_source(text):
    """Point the template's reference_camera at a source assembly removes.

    Every shipped template anchors on a baked-in source, so the reconciliation
    is unreachable through them — which is exactly how it could rot unnoticed.
    """
    text = _insert_source(text, _HEAD_IPHONE_SOURCE)
    for old, new in (
        # Both the top-level key and the calibration check's own.
        (
            "reference_camera: right_tracking_camera",
            "reference_camera: head_iphone_camera",
        ),
        (
            "cameras: [right_tracking_camera, left_tracking_camera]",
            "cameras: [right_tracking_camera, left_tracking_camera, "
            "head_iphone_camera]",
        ),
    ):
        assert old in text, old
        text = text.replace(old, new)
    return text


def test_a_reference_camera_that_assembly_removes_is_repointed(tmp_path):
    """The post-condition of the stale-reference hazard, exercised rather than
    assumed.

    RecorderManager._validate_reference_camera raises at construction when
    reference_camera names no source, aborting the session before the first
    episode. The assembler must never be able to write such a file.

    The auto-pick runs for every rig and overwrites the template's value
    outright, so a stale name is gone before `_reconcile_reference_camera` is
    even reached. The guarantee asserted here is the one that matters -- no
    file leaves this method naming a camera the session does not record.
    """
    repo = _temp_repo(tmp_path, _anchored_on_a_removable_source)
    cfg = _load(ConfigAssembler(repo).assemble(_session(), _devices()))

    names = _source_names(cfg)
    assert "head_iphone_camera" not in names, "assembly should remove it"
    assert cfg["reference_camera"] in names
    # The anchor the rule picks for the V4 rig with no head OAK: the right
    # tracking camera, in both places RecorderManager reads.
    assert cfg["reference_camera"] == "right_tracking_camera"
    params = _calibration_params(cfg)
    assert params["reference_camera"] == "right_tracking_camera"
    assert not set(params.get("cameras") or []) - names


def test_adding_a_head_iphone_does_not_move_the_reference_camera():
    """The head OAK is the top of the reference priority, and the head iPhone
    deliberately is not: it is a tracking camera in a head mount, not the fixed
    workspace camera the world frame wants. An optional add-on must not
    silently change what every extrinsic is relative to, and this is also the
    line between the two head devices. See
    `ConfigAssembler._HEAD_REFERENCE_NAMES`."""
    assembler = ConfigAssembler(REPO)
    plain = _load(assembler.assemble(_session(), _devices()))
    with_head = _load(
        assembler.assemble(
            _session(),
            _devices(
                optional=(
                    OptionalDeviceAssignment(
                        "head_iphone", "head_iphone_camera", "iphone16p-ry"
                    ),
                ),
            ),
        )
    )
    assert with_head["reference_camera"] == plain["reference_camera"]


# --------------------------------------------------------------- writing


def test_write_is_atomic(monkeypatch):
    """A failure mid-write must leave the previous file intact."""
    assembler = ConfigAssembler(REPO)
    session = _session()
    good = assembler.assemble(session, _devices())
    original = good.read_bytes()

    from station_ui.core import config_assembler as mod

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(RuntimeError):
        assembler.assemble(session, _devices())
    assert good.read_bytes() == original


def test_a_failed_write_leaves_no_temporary_file(monkeypatch):
    assembler = ConfigAssembler(REPO)
    session = _session()
    out = assembler.assemble(session, _devices())

    from station_ui.core import config_assembler as mod

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(RuntimeError):
        assembler.assemble(session, _devices())
    assert not list(out.parent.glob("*.tmp"))


def test_added_keys_and_sources_stay_behind_their_section_banner():
    """ruamel attaches a comment to the key it follows, so the last entry of a
    block owns the banner that introduces the *next* one. Appending naively
    files new keys under the wrong heading in the config the operator reads."""
    out = ConfigAssembler(REPO).assemble(
        _session(),
        _devices(
            optional=(
                OptionalDeviceAssignment(
                    "bottom_oak", "bottom_oak_camera", "oak-W97-52", "table"
                ),
            ),
        ),
    )
    text = out.read_text()
    assert text.index("session_uid:") < text.index("# General recording settings")
    assert text.index("data_dir:") < text.index("# Video sources")
    assert text.index("bottom_oak_camera") < text.index(
        "# Identifies the world-frame camera"
    )
    assert text.index("# Identifies the world-frame camera") < text.index(
        "reference_camera:"
    )


def test_a_nested_block_is_never_split_by_a_carried_comment():
    # meta.hardware is a nested map; a banner parked on it renders between
    # "hardware:" and its first child.
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    lines = out.read_text().splitlines()
    hardware = lines.index("  hardware:")
    assert lines[hardware + 1].strip().startswith("left_umi_gripper:")


def test_repeated_values_do_not_serialize_as_yaml_anchors():
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    text = out.read_text()
    assert "&id00" not in text
    assert "*id00" not in text


def test_the_generated_file_is_plain_loadable_yaml():
    yaml = pytest.importorskip("yaml")
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    cfg = yaml.safe_load(out.read_text())
    assert cfg["meta"]["session_uid"] == "91e98c89fccb"


def test_output_path_is_stable_for_a_preset():
    assembler = ConfigAssembler(REPO)
    session = _session()
    assert assembler.output_path(session) == assembler.assemble(session, _devices())
    assert assembler.output_path(session).parts[-2:] == (
        "generated",
        "bimanual_v4.yaml",
    )


def test_assembling_twice_is_byte_identical():
    assembler = ConfigAssembler(REPO)
    first = assembler.assemble(_session(), _devices()).read_bytes()
    second = assembler.assemble(_session(), _devices()).read_bytes()
    assert first == second


# Verify parsed-template cache isolation and invalidation so callers
# cannot receive a different or mutated document.


def test_a_reader_never_receives_the_cached_document_itself(tmp_path):
    """Two readers of one file must not share a ruamel node.

    `_optional_source_specs` relies on this: comment tokens live on the nodes,
    so one caller's edit would rewrite what the next caller reads.
    """
    path = tmp_path / "t.yaml"
    path.write_text("a: 1\nb: [1, 2]\n")

    first = mod._load_yaml(path)
    second = mod._load_yaml(path)
    assert second == first
    assert second is not first

    second["a"] = 999
    assert mod._load_yaml(path)["a"] == 1, "a caller's edit reached the cache"


def test_an_edit_of_the_same_length_is_still_seen(tmp_path):
    """The cache is keyed on the text, not on (path, mtime, size).

    Rewrite a file with content of the same length fast enough that both writes
    land in one mtime tick -- which takes no effort at all -- and a stat-keyed
    cache would serve the previous parse. Keying on the text is what makes the
    question go away, and this is the test that says so.
    """
    path = tmp_path / "t.yaml"
    path.write_text("a: 22222\nb: [1, 2]\n")
    assert mod._load_yaml(path)["a"] == 22222

    size_before = path.stat().st_size
    path.write_text("a: 33333\nb: [1, 2]\n")
    assert path.stat().st_size == size_before, "this no longer exercises the case"

    assert mod._load_yaml(path)["a"] == 33333, "served a stale parse"


def test_the_cache_cannot_grow_without_bound(tmp_path):
    """Every test here builds its own copy of the config tree in its own tmpdir,
    so an unbounded cache would hold a parsed document per temp directory for
    the whole session."""
    for index in range(mod._parsed_once.cache_info().maxsize * 3):
        path = tmp_path / f"g{index}.yaml"
        path.write_text(f"n: {index}\n")
        mod._load_yaml(path)

    info = mod._parsed_once.cache_info()
    assert info.currsize <= info.maxsize


def test_camera_index_zero_is_a_device_identifier():
    assert (
        mod.source_device({"type": "uvc_camera", "params": {"camera_index": 0}}) == "0"
    )
