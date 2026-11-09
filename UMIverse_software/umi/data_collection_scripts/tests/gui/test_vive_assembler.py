# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the console learns about a Vive rig by reading its template.

Identification is a type question (`type: vive_tracker`); which arm a source
drives is a naming question, answered by the rule
`umi/common/utility/tracking.py` already owns. Nothing here re-implements that
rule -- another copy of it is exactly what that module's docstring exists to
prevent.
"""

import pytest

ruamel = pytest.importorskip("ruamel.yaml")

import pathlib  # noqa: E402

from station_ui.core import config_assembler as mod  # noqa: E402
from station_ui.core.config_assembler import (  # noqa: E402
    ConfigAssembler,
    DeviceSpec,
    SessionSpec,
)
from station_ui.core.setup_presets import (  # noqa: E402
    POSE_TRACKER_VIVE,
    SETUP_PRESETS,
    get_preset,
    template_path,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _template(preset_id):
    path = template_path(get_preset(preset_id), REPO)
    return ruamel.YAML().load(path.read_text())


def test_a_vive_template_declares_one_source_per_arm():
    cfg = _template("bimanual_v4_vive")
    sources = ConfigAssembler._requirements(cfg).vive_sources

    assert [s.arm for s in sources] == ["left", "right"]
    assert [s.name for s in sources] == ["left_tracking_pose", "right_tracking_pose"]
    assert [s.tracker_key for s in sources] == ["left_knee", "right_knee"]
    # Both sources name one endpoint. That is what makes "one client per rig"
    # correct, and what lets the console offer a single Server box.
    assert {s.endpoint for s in sources} == {"tcp://127.0.0.1:5555"}


def test_a_rig_with_no_vive_sources_declares_none():
    """Every non-Vive preset must come back empty -- this is the single switch
    the config section and the calibration step both key on."""
    others = [p.id for p in SETUP_PRESETS if p.pose_tracker != POSE_TRACKER_VIVE]
    assert others, "no non-Vive preset left to check"
    for preset_id in others:
        assert ConfigAssembler._requirements(_template(preset_id)).vive_sources == ()


def test_vive_sources_do_not_leak_into_gripper_arms():
    """`gripper_arms` filters on type and the _umi_gripper suffix, so this holds
    with no change -- asserted so a future edit to either filter cannot
    silently make a tracker look like a gripper."""
    requirements = ConfigAssembler._requirements(_template("bimanual_v4_vive"))
    assert requirements.gripper_arms == ("left", "right")
    assert len(requirements.vive_sources) == 2


def test_a_tracker_that_names_no_arm_is_skipped():
    """A head-mounted puck is not an arm. `tracking_device_to_arm` returns None
    for `head_*`, and this must drop the source rather than inventing an arm
    called "head"."""
    cfg = _template("bimanual_v4_vive")
    head = ruamel.YAML().load(
        "name: head_tracking_pose\n"
        "type: vive_tracker\n"
        "params:\n"
        "  endpoint: tcp://x:1\n"
        "  tracker_key: head_puck\n"
    )
    cfg["numeric_sources"].append(head)

    arms = [s.arm for s in ConfigAssembler._requirements(cfg).vive_sources]
    assert arms == ["left", "right"]


@pytest.fixture(autouse=True)
def _isolated_output(tmp_path, monkeypatch):
    """Assemble into tmp_path, never into the station's own state dir.

    output_path() writes under the station's live state dir (`~/.umi_station`),
    which is the config the console has applied and reopens on the next visit --
    so a suite left unredirected would arm the bench it ran on.
    """
    monkeypatch.setattr(mod, "GENERATED_DIR", (str(tmp_path), "generated"))


def _session(**kw):
    base = dict(
        preset_id="bimanual_v4_vive",
        operator_name="Test Operator",
        task_name="plane_assembly_v9",
        session_uid="91e98c89fccb",
        data_dir="output/raw/plane_assembly_v9/2026-08-16_x_91e98c89fccb",
    )
    base.update(kw)
    return SessionSpec(**base)


def _cameras(preset_id="bimanual_v4_vive"):
    """One device per block of a template, so assemble is not rejected.

    Per block, not per stream: every stream of one OAK block must name the same
    device and no two blocks may share one, or `_reject_shared_devices` refuses
    the form before any of this reaches the Vive sources. The single-stream
    fallback is the one `_requirements` applies -- a USB camera or an iPhone
    carries its name on the source, not under `streams`. A USB camera is
    identified by its V4L2 index, so it gets the block's own number rather than
    a device name, which the assembler rightly refuses.
    """
    cfg = _template(preset_id)
    cameras = {}
    for index, source in enumerate(cfg.get("video_sources") or []):
        names = [s["name"] for s in source.get("streams") or [] if s.get("name")]
        uvc = source.get("type") == "uvc_camera"
        for name in names or ([source["name"]] if source.get("name") else []):
            cameras[name] = str(index) if uvc else f"fixture-device-{index}"
    return cameras


def _devices(**kw):
    base = dict(
        grippers={"left": "UMI-V4-0001", "right": "UMI-V4-0002"},
        required_cameras=_cameras(),
        optional=(),
    )
    base.update(kw)
    return DeviceSpec(**base)


def _written(out):
    cfg = ruamel.YAML().load(pathlib.Path(out).read_text())
    return {
        entry["name"]: entry.get("params") or {}
        for entry in cfg["numeric_sources"]
        if entry.get("type") == "vive_tracker"
    }


def test_an_operators_endpoint_and_keys_reach_both_sources():
    out = ConfigAssembler(REPO).assemble(
        _session(),
        _devices(
            vive_endpoint="tcp://10.20.30.40:5555",
            vive_keys={"left": "left_shoulder", "right": "right_shoulder"},
        ),
    )
    params = _written(out)

    assert params["left_tracking_pose"]["endpoint"] == "tcp://10.20.30.40:5555"
    assert params["right_tracking_pose"]["endpoint"] == "tcp://10.20.30.40:5555"
    assert params["left_tracking_pose"]["tracker_key"] == "left_shoulder"
    assert params["right_tracking_pose"]["tracker_key"] == "right_shoulder"


def test_an_empty_field_leaves_the_templates_own_value_standing():
    """Empty means "the template's value stands", not "blank". Every Vive
    template ships a working endpoint and key pair, so the bench a template was
    written for needs zero clicks -- which is the deliberate difference from a
    gripper slot, where `_reject_partial_form` refuses an empty box because the
    placeholder serial would otherwise be recorded as though it were real
    hardware."""
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    params = _written(out)

    assert params["left_tracking_pose"]["endpoint"] == "tcp://127.0.0.1:5555"
    assert params["left_tracking_pose"]["tracker_key"] == "left_knee"
    assert params["right_tracking_pose"]["tracker_key"] == "right_knee"


def test_one_arms_key_can_be_set_without_touching_the_other():
    out = ConfigAssembler(REPO).assemble(
        _session(), _devices(vive_keys={"left": "left_shoulder"})
    )
    params = _written(out)

    assert params["left_tracking_pose"]["tracker_key"] == "left_shoulder"
    assert params["right_tracking_pose"]["tracker_key"] == "right_knee"


def test_the_other_tuning_params_are_left_alone():
    """poll_hz / timeout_s / max_age_s are measured values with a long comment
    above them. The console writes two keys and touches nothing else."""
    out = ConfigAssembler(REPO).assemble(
        _session(), _devices(vive_endpoint="tcp://10.20.30.40:5555")
    )
    left = _written(out)["left_tracking_pose"]

    assert left["poll_hz"] == 90
    assert left["timeout_s"] == 0.25
    assert left["max_age_s"] == 0.15


def test_two_arms_on_one_puck_are_refused():
    """One puck cannot drive two arms, and nothing downstream would notice.

    Both sources would record the same pose: two plausible traces that move,
    so pose_movement, pose_jumps, consecutive_movement and the arm-steady
    pairing all pass and the episode goes out green with one arm carrying the
    other's trajectory. The tracker check would stall rather than say so -- so,
    exactly as for a shared gripper, the only place this can be caught is before
    the file is written.
    """
    with pytest.raises(ValueError) as excinfo:
        ConfigAssembler(REPO).assemble(
            _session(),
            _devices(vive_keys={"left": "left_knee", "right": "left_knee"}),
        )

    message = str(excinfo.value)
    assert "left_knee" in message
    assert "left_tracking_pose" in message
    assert "right_tracking_pose" in message


def test_an_arm_pointed_at_the_other_arms_UNTOUCHED_key_is_refused():
    """The likelier half of the mistake: the operator fixes one box.

    Only `right` is typed, and it is typed as the key the left source still
    holds from the template. Comparing what the operator typed against what the
    other arm will actually record -- not against what they typed there -- is
    what catches it; a check that only compared the two typed boxes would see
    one value and pass.
    """
    with pytest.raises(ValueError) as excinfo:
        ConfigAssembler(REPO).assemble(
            _session(), _devices(vive_keys={"right": "left_knee"})
        )

    assert "left_knee" in str(excinfo.value)


def test_two_arms_on_two_pucks_are_not_refused():
    """The check must not be over-broad: distinct keys are the normal case, and
    a rig that renamed both pucks has to stay applicable."""
    out = ConfigAssembler(REPO).assemble(
        _session(),
        _devices(vive_keys={"left": "left_shoulder", "right": "right_shoulder"}),
    )
    params = _written(out)

    assert params["left_tracking_pose"]["tracker_key"] == "left_shoulder"
    assert params["right_tracking_pose"]["tracker_key"] == "right_shoulder"


def test_a_source_that_ships_no_params_block_gets_one():
    """`entry.get("params")` is None for a vive source declaring no `params:`,
    and assigning into None is a TypeError -- a 500 on the page rather than a
    written config. Driven directly, because every template in the tree ships
    the block: only a hand-edited station file reaches this."""
    cfg = _template("bimanual_v4_vive")
    bare = ruamel.YAML().load("name: left_tracking_pose\ntype: vive_tracker\n")
    cfg["numeric_sources"] = [
        bare if entry.get("name") == "left_tracking_pose" else entry
        for entry in cfg["numeric_sources"]
    ]

    ConfigAssembler._apply_vive(
        cfg,
        DeviceSpec(
            vive_endpoint="tcp://10.20.30.40:5555",
            vive_keys={"left": "left_shoulder"},
        ),
    )

    assert bare["params"]["endpoint"] == "tcp://10.20.30.40:5555"
    assert bare["params"]["tracker_key"] == "left_shoulder"


def test_a_non_vive_rig_is_unaffected_by_vive_fields():
    """_apply_vive iterates the template's vive_sources, so a rig that declares
    none is a no-op even when the draft carries values."""
    out = ConfigAssembler(REPO).assemble(
        _session(preset_id="bimanual_v4"),
        DeviceSpec(
            grippers={"left": "UMI-V4-0001", "right": "UMI-V4-0002"},
            required_cameras=_cameras("bimanual_v4"),
            optional=(),
            vive_endpoint="tcp://ignored:1",
            vive_keys={"left": "ignored"},
        ),
    )
    cfg = ruamel.YAML().load(pathlib.Path(out).read_text())
    assert not [e for e in cfg["numeric_sources"] if e.get("type") == "vive_tracker"]


def test_the_endpoint_and_keys_round_trip_through_the_written_file():
    """Reopen. This is also why no new state file is needed: `session_store`
    rebuilds the draft after a restart by re-reading the generated config
    through this same `hydrate`, so an applied endpoint comes back on its own."""
    out = ConfigAssembler(REPO).assemble(
        _session(),
        _devices(
            vive_endpoint="tcp://10.20.30.40:5555",
            vive_keys={"left": "left_shoulder", "right": "right_shoulder"},
        ),
    )

    from station_ui import config_form
    from station_ui.session import SessionDraft

    draft = SessionDraft()
    config_form.hydrate(draft, ruamel.YAML().load(pathlib.Path(out).read_text()))

    assert draft.vive_endpoint == "tcp://10.20.30.40:5555"
    assert draft.vive_keys == {"left": "left_shoulder", "right": "right_shoulder"}


def test_reopening_a_non_vive_config_leaves_the_vive_fields_empty():
    from station_ui import config_form
    from station_ui.session import SessionDraft

    draft = SessionDraft()
    draft.vive_endpoint = "tcp://stale:1"
    draft.vive_keys = {"left": "stale"}
    config_form.hydrate(draft, _template("bimanual_v4"))

    assert draft.vive_endpoint == ""
    assert draft.vive_keys == {}


def test_hydrate_never_raises_on_a_malformed_vive_source():
    """`session_store.restore` wraps hydrate in a bare `except Exception` that
    discards the entire restored draft -- so a raise here loses the session uid
    and the recording date, forking a half-recorded session across two
    folders. Every read in the Vive block is total."""
    from station_ui import config_form
    from station_ui.session import SessionDraft

    cfg = {
        "numeric_sources": [
            {"type": "vive_tracker"},  # no name at all
            {"type": "vive_tracker", "name": "left_tracking_pose"},  # no params
            {"type": "vive_tracker", "name": "junk", "params": None},
            # A nonempty string params value is truthy but not a mapping; hydration
            # must tolerate this malformed source without calling str.get.
            {
                "type": "vive_tracker",
                "name": "left_tracking_pose",
                "params": "tcp://10.1.1.1:5555",
            },
        ]
    }
    draft = SessionDraft()
    config_form.hydrate(draft, cfg)  # must not raise

    assert draft.vive_keys == {}


def test_meta_names_vive_as_the_tracking_source_of_both_arms():
    """The whole point of the preset: pose comes from a puck, not a phone. The
    two rigs are otherwise identical in `meta` apart from `setup_preset`, so
    without this a consumer asking "what produced this pose" would have to know
    the preset ids."""
    out = ConfigAssembler(REPO).assemble(_session(), _devices())
    cfg = ruamel.YAML().load(pathlib.Path(out).read_text())

    assert cfg["meta"]["tracking_source"] == {
        "head": "notracker",
        "left": "vive",
        "right": "vive",
    }
