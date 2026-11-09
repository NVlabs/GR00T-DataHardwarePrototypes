# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session metadata must survive source expansion and reach the recorded session.

``create_video_recorders`` / ``create_numeric_recorders`` rebuild every declared
source into a *flat* ``entry`` dict, and ``record_data.py`` overwrites
``cfg["video_sources"]`` with those expanded lists **before**
``RecorderManager.__init__`` writes the session ``config.yaml``. Any key not
copied into ``entry`` is silently dropped before it reaches the session
directory, the data check, alignment or the LeRobot conversion.

Two pieces of metadata are covered here:

* ``placement`` — descriptive physical location of a camera or device.
* ``meta.session_uid`` — the session's collision-proof identity.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import zarr

from umi.common.recording.record_manager import RecorderManager
from umi.common.utility.recording_utils import (
    create_numeric_recorders,
    create_video_recorders,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_name(expanded):
    return {entry["name"]: entry for entry in expanded}


def _mock_source(device_name: str) -> MagicMock:
    source = MagicMock()
    source.device_name = device_name
    return source


def _iphone_source(**overrides):
    """A non-oak source block, shaped like what the assembler emits for a head
    iPhone."""
    source = {
        "type": "iphone_camera",
        "name": "head_iphone_camera",
        "frame_type": "RGBDCameraFrame",
        "latency": 0.125,
        "record_fps": 30,
        "stream_fps": 90,
        "placement": "neck",
        "params": {
            "iphone_device_name": "iphone16p-ry",
            "camera_resolution": [720, 960],
        },
    }
    source.update(overrides)
    return source


def _oak_source(**overrides):
    """An oak_device block carrying two streams."""
    source = {
        "type": "oak_device",
        "oak_device_name": "oak-FFC-03",
        "fps": 45,
        "latency": 0.0823,
        "streams": [
            {
                "name": "left_trigger_camera",
                "socket": "CAM_A",
                "resolution": [640, 400],
            },
            {
                "name": "left_umi_camera",
                "socket": "CAM_B",
                "resolution": [640, 400],
            },
        ],
    }
    source.update(overrides)
    return source


def _expand_video(source_configs, record_path):
    """Run create_video_recorders with hardware construction stubbed out."""
    with (
        patch("umi.common.utility.recording_utils.create_video_source") as create,
        patch("umi.common.utility.recording_utils.VideoRecorder"),
    ):

        def side_effect(cfg):
            if cfg["type"] == "oak_device":
                return [_mock_source(s["name"]) for s in cfg["streams"]]
            return [_mock_source(cfg["name"])]

        create.side_effect = side_effect
        _, _, expanded = create_video_recorders(
            video_source_configs=source_configs,
            record_path=record_path,
        )
    return expanded


# ---------------------------------------------------------------------------
# placement through the video expansion
# ---------------------------------------------------------------------------


class TestVideoExpansionMetadata:
    def test_placement_on_a_non_oak_source_survives_expansion(self, tmp_path):
        expanded = _expand_video([_iphone_source()], str(tmp_path))
        assert _by_name(expanded)["head_iphone_camera"]["placement"] == "neck"

    def test_missing_placement_is_omitted_not_null(self, tmp_path):
        source = _iphone_source()
        source.pop("placement")
        entry = _by_name(_expand_video([source], str(tmp_path)))["head_iphone_camera"]
        # Absent, not None: old and new sessions must stay comparable.
        assert "placement" not in entry

    def test_device_level_placement_reaches_every_oak_stream(self, tmp_path):
        """A key on the oak_device block falls back onto each of its streams."""
        expanded = _by_name(
            _expand_video([_oak_source(placement="left_arm")], str(tmp_path))
        )
        assert expanded["left_trigger_camera"]["placement"] == "left_arm"
        assert expanded["left_umi_camera"]["placement"] == "left_arm"

    def test_stream_level_placement_overrides_the_device_level_one(self, tmp_path):
        source = _oak_source(placement="left_arm")
        source["streams"][0]["placement"] = "left_trigger"
        expanded = _by_name(_expand_video([source], str(tmp_path)))
        assert expanded["left_trigger_camera"]["placement"] == "left_trigger"
        assert expanded["left_umi_camera"]["placement"] == "left_arm"

    def test_no_metadata_anywhere_adds_no_keys(self, tmp_path):
        for entry in _expand_video([_oak_source()], str(tmp_path)):
            assert "placement" not in entry
            assert "sensor_module" not in entry


# ---------------------------------------------------------------------------
# placement through the numeric expansion
# ---------------------------------------------------------------------------


def _sensor_source(**overrides):
    source = {
        "type": "mock_sensor",
        "name": "left_umi_gripper",
        "record_fps": 30,
        "stream_fps": 60,
        "latency": 0.01,
        "placement": "left",
        "params": {"num_channels": 3},
    }
    source.update(overrides)
    return source


class TestNumericExpansionMetadata:
    def test_placement_survives_expansion(self, tmp_path):
        _, _, expanded = create_numeric_recorders(
            numeric_source_configs=[_sensor_source()],
            record_path=str(tmp_path),
        )
        assert _by_name(expanded)["left_umi_gripper"]["placement"] == "left"

    def test_missing_placement_is_omitted_not_null(self, tmp_path):
        source = _sensor_source()
        source.pop("placement")
        _, _, expanded = create_numeric_recorders(
            numeric_source_configs=[source],
            record_path=str(tmp_path),
        )
        assert "placement" not in _by_name(expanded)["left_umi_gripper"]


# ---------------------------------------------------------------------------
# meta.session_uid
# ---------------------------------------------------------------------------

_UID = "91e98c89fccb"


def _assembled_config(**meta_overrides):
    """A config shaped like ConfigAssembler output (carries setup_preset)."""
    meta = {
        "operator_name": "Test Operator",
        "task_name": "plane_assembly_v9",
        "setup_preset": "bimanual_v4",
        "operation_mode": "bimanual",
        "gripper_pipeline": "v4",
        "session_uid": _UID,
    }
    meta.update(meta_overrides)
    return {"meta": meta, "video_sources": [], "numeric_sources": []}


def _manager(record_path, config):
    return RecorderManager(recorders=[], record_path=str(record_path), config=config)


class TestSessionUid:
    def test_session_uid_reaches_the_dataset_zarr_attrs(self, tmp_path):
        _manager(tmp_path, _assembled_config())
        store = zarr.open(str(tmp_path), mode="r")
        assert store.attrs["session_uid"] == _UID

    def test_missing_session_uid_raises(self, tmp_path):
        config = _assembled_config()
        config["meta"].pop("session_uid")
        with pytest.raises(ValueError, match="session_uid"):
            _manager(tmp_path, config)

    @pytest.mark.parametrize(
        "bad",
        ["", None, "91E98C89FCCB", "91e98c89fcc", "91e98c89fccbb", "zzzzzzzzzzzz"],
    )
    def test_malformed_session_uid_raises_on_every_path(self, tmp_path, bad):
        """A malformed uid is never intentional — reject it even on a legacy config."""
        with pytest.raises(ValueError, match="session_uid"):
            _manager(tmp_path, _assembled_config(session_uid=bad))

    def test_validation_is_not_swallowed_by_save_meta_data(self, tmp_path):
        """The check must run outside _save_meta_data's bare ``except``.

        _save_meta_data is called from __init__ inside a try/except that only
        prints, so a check living in there would be invisible. Neutering
        _save_meta_data entirely must not neuter the validation.
        """
        config = _assembled_config()
        config["meta"].pop("session_uid")
        with patch.object(RecorderManager, "_save_meta_data", lambda _: None):
            with pytest.raises(ValueError, match="session_uid"):
                _manager(tmp_path, config)

    def test_legacy_config_without_a_session_identity_block_still_constructs(
        self, tmp_path
    ):
        """Hand-written configs predate session_uid and must keep working.

        Some carry a meta block with no session identity, some no meta block
        at all.
        """
        # One session directory per config, so each construction starts from
        # an empty session folder.
        for index, config in enumerate(
            [{"meta": {"task_name": "cube"}}, {"video_sources": []}, None]
        ):
            path = tmp_path / f"session_{index}"
            path.mkdir()
            _manager(path, config)

    def test_no_session_uid_attr_is_written_for_a_legacy_config(self, tmp_path):
        _manager(tmp_path, {"meta": {"task_name": "cube"}})
        store = zarr.open(str(tmp_path), mode="r")
        assert "session_uid" not in dict(store.attrs)

    def test_session_uid_is_written_alongside_the_existing_meta_attrs(self, tmp_path):
        _manager(tmp_path, _assembled_config(HW_version="v4"))
        attrs = dict(zarr.open(str(tmp_path), mode="r").attrs)
        assert attrs["operator_name"] == "Test Operator"
        assert attrs["task_name"] == "plane_assembly_v9"
        assert attrs["HW_version"] == "v4"
        assert attrs["session_uid"] == _UID
        assert "commit_id" in attrs


class TestTrackingSource:
    """`meta.tracking_source` says what produced each slot's pose.

    It is written by the station console, which knows the rig, and read here
    rather than re-derived: `create_video_recorders` has already flattened
    every source block by the time this config reaches the recorder, so a
    consumer asking the recorded sources would be asking the wrong copy.
    """

    _BLOCK = {"head": "none", "left": "vive", "right": "vive"}

    def test_tracking_source_reaches_the_dataset_zarr_attrs(self, tmp_path):
        _manager(tmp_path, _assembled_config(tracking_source=dict(self._BLOCK)))
        attrs = dict(zarr.open(str(tmp_path), mode="r").attrs)
        assert attrs["tracking_source"] == self._BLOCK

    def test_a_malformed_block_costs_nothing_but_the_attr(self, tmp_path):
        """`_save_meta_data` runs inside a bare `except` that only prints, so a
        raise here silently skips the `discarded_reasons` seed and the QC sync
        that follow it. A hand-edited scalar must cost this one attr alone."""
        _manager(tmp_path, _assembled_config(tracking_source="vive"))
        attrs = dict(zarr.open(str(tmp_path), mode="r").attrs)
        assert "tracking_source" not in attrs
        assert attrs["discarded_reasons"] == {}

    def test_no_attr_is_written_for_a_config_that_declares_none(self, tmp_path):
        """Hand-written configs reach this constructor too, and never carry
        the key. Absent must stay absent rather than becoming an
        empty block that reads as "recorded no pose"."""
        _manager(tmp_path, _assembled_config())
        assert "tracking_source" not in dict(zarr.open(str(tmp_path), mode="r").attrs)


# ---------------------------------------------------------------------------
# End to end: expansion output is what actually lands in the session config
# ---------------------------------------------------------------------------


def test_metadata_reaches_the_saved_session_config(tmp_path):
    """The whole point: what record_data.py writes to disk keeps the keys."""
    import yaml

    record_path = tmp_path / "session"
    record_path.mkdir()

    cfg = _assembled_config()
    cfg["video_sources"] = [_iphone_source(), _oak_source()]
    cfg["numeric_sources"] = [_sensor_source()]

    expanded_video = _expand_video(cfg["video_sources"], str(record_path))
    _, _, expanded_numeric = create_numeric_recorders(
        numeric_source_configs=cfg["numeric_sources"],
        record_path=str(record_path),
    )
    # Exactly what record_data.py does before building the RecorderManager.
    cfg["video_sources"] = expanded_video
    cfg["numeric_sources"] = expanded_numeric

    _manager(record_path, cfg)

    with open(os.path.join(record_path, "config.yaml")) as f:
        saved = yaml.safe_load(f)

    video = _by_name(saved["video_sources"])
    assert video["head_iphone_camera"]["placement"] == "neck"
    assert _by_name(saved["numeric_sources"])["left_umi_gripper"]["placement"] == "left"
    assert saved["meta"]["session_uid"] == _UID
