# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which camera recorded this episode, readable from the episode alone.

The session config names each arm's camera (`meta.hardware.<arm>.camera`).
That answers the question when the config is at hand, but episodes are
quarantined and re-examined one at a time, often long after the fact and
without the config beside them -- so each episode records its camera identity
next to its gripper identity.

Two decisions worth stating, because both could reasonably have gone the other
way.

The stored name is the raw spelling from the config, not a canonical identity.
`canonical_device_identity` falls back to the spelling it was given when the
registry cannot be loaded, so canonicalising here would freeze a value that
depends on whether a registry fetch happened to succeed at record time. A
permanent record must not have that property. The serial is the canonical
identity, and it comes from the registry's own name -> serial table.

An unresolvable camera keeps its name and omits the serial, which differs from
the gripper writer beside it -- that one drops the source entirely. Deliberate:
absence is ambiguous (no camera? unregistered? lookup failed?), while a name
with no serial says exactly what is and is not known. Fabricating a serial from
the name is the one option ruled out, since that is the non-deterministic value
above wearing the canonical field's clothes.
"""

import pathlib
import types

import pytest
import yaml
import zarr

from umi.common.recording.constants import PENDING_SUFFIX
from umi.common.recording.record_manager import RecorderManager

OAK_SERIAL = "1944301000000000AC"
IPHONE_UDID = "00008140-00000000000000A5"


def _registry():
    return {
        "version": 1,
        "oak_devices": {"oak-W97-07": OAK_SERIAL},
        "iphone_devices": {"iphone15p-04": IPHONE_UDID},
        "umi_v4_grippers": {},
        "gripper_cameras": {},
    }


@pytest.fixture(autouse=True)
def _fleet(monkeypatch):
    import umi.utils.device_manager as manager
    import umi.utils.device_registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", _registry(), raising=False)
    for name in ("IPHONE_DEVICES", "OAK_DEVICES"):
        getattr(manager, name).invalidate()
    yield
    monkeypatch.setattr(registry, "_REGISTRY", None, raising=False)


def _recorder(tmp_path, video_sources):
    path = tmp_path / "session.zarr"
    store = zarr.open(str(path), mode="a")
    store.create_group(f"episode_0{PENDING_SUFFIX}")
    return types.SimpleNamespace(
        record_path=str(path),
        episode_id=0,
        verbose=False,
        config={"video_sources": video_sources},
    )


def _attrs(recorder):
    store = zarr.open(recorder.record_path, mode="a")
    return dict(store[f"episode_0{PENDING_SUFFIX}"].attrs)


def _source(name, key, value, source_type="oak_camera"):
    return {"name": name, "type": source_type, "params": {key: value}}


class TestTheEpisodeCarriesItsCameras:
    def test_an_oak_camera_reaches_the_attrs(self, tmp_path):
        recorder = _recorder(
            tmp_path, [_source("left_umi_camera", "oak_device_name", "oak-W97-07")]
        )

        RecorderManager._save_camera_serial_data(recorder)

        assert _attrs(recorder)["camera_serial"] == {
            "left_umi_camera": {
                "device_name": "oak-W97-07",
                "serial": OAK_SERIAL,
            }
        }

    def test_every_camera_type_is_covered(self, tmp_path):
        recorder = _recorder(
            tmp_path,
            [
                _source("umi", "oak_device_name", "oak-W97-07"),
                _source("track", "iphone_device_name", "iphone15p-04", "iphone_camera"),
            ],
        )

        RecorderManager._save_camera_serial_data(recorder)

        stored = _attrs(recorder)["camera_serial"]
        assert set(stored) == {"umi", "track"}
        assert stored["track"]["serial"] == IPHONE_UDID
        assert all(entry.get("serial") for entry in stored.values())

    def test_an_episode_can_be_audited_without_its_config(self, tmp_path):
        """The whole point. A quarantined episode is re-examined alone, and the
        attrs must name the board on their own."""
        recorder = _recorder(
            tmp_path, [_source("left_umi_camera", "oak_device_name", "oak-W97-07")]
        )

        RecorderManager._save_camera_serial_data(recorder)
        stored = _attrs(recorder)["camera_serial"]["left_umi_camera"]

        assert stored["serial"] == OAK_SERIAL
        assert stored["device_name"] == "oak-W97-07"


class TestTheStoredNameIsTheRawSpelling:
    def test_the_operators_spelling_survives_verbatim(self, tmp_path):
        """Not canonicalised. `canonical_device_identity` falls back to the raw
        spelling when the registry cannot load, so canonicalising here would
        freeze a value that depends on whether a registry fetch succeeded."""
        recorder = _recorder(
            tmp_path, [_source("left_umi_camera", "oak_device_name", "OAK-w97-07")]
        )

        RecorderManager._save_camera_serial_data(recorder)
        stored = _attrs(recorder)["camera_serial"]["left_umi_camera"]

        assert stored["device_name"] == "OAK-w97-07"

    def test_the_serial_is_still_resolved_from_a_drifted_spelling(self, tmp_path):
        """The name is provenance; the serial is identity. A case-drifted name
        must still resolve, or the identity field is lost to a typo."""
        recorder = _recorder(
            tmp_path, [_source("left_umi_camera", "oak_device_name", "OAK-w97-07")]
        )

        RecorderManager._save_camera_serial_data(recorder)

        assert _attrs(recorder)["camera_serial"]["left_umi_camera"]["serial"] == (
            OAK_SERIAL
        )


class TestAnUnresolvableCameraSaysSo:
    def test_the_name_is_kept_and_the_serial_omitted(self, tmp_path):
        recorder = _recorder(
            tmp_path, [_source("left_umi_camera", "oak_device_name", "oak-UNKNOWN-99")]
        )

        RecorderManager._save_camera_serial_data(recorder)
        stored = _attrs(recorder)["camera_serial"]["left_umi_camera"]

        assert stored["device_name"] == "oak-UNKNOWN-99"
        assert "serial" not in stored

    def test_the_name_is_never_written_into_the_serial_field(self, tmp_path):
        """The one option ruled out: a serial that is sometimes a name is worse
        than no serial, because nothing downstream can tell which it got."""
        recorder = _recorder(
            tmp_path, [_source("left_umi_camera", "oak_device_name", "oak-UNKNOWN-99")]
        )

        RecorderManager._save_camera_serial_data(recorder)
        stored = _attrs(recorder)["camera_serial"]["left_umi_camera"]

        assert stored.get("serial") != "oak-UNKNOWN-99"

    def test_one_unresolvable_camera_does_not_lose_the_others(self, tmp_path):
        recorder = _recorder(
            tmp_path,
            [
                _source("good", "oak_device_name", "oak-W97-07"),
                _source("bad", "oak_device_name", "oak-UNKNOWN-99"),
            ],
        )

        RecorderManager._save_camera_serial_data(recorder)
        stored = _attrs(recorder)["camera_serial"]

        assert stored["good"]["serial"] == OAK_SERIAL
        assert "serial" not in stored["bad"]


class TestItStaysQuietWhenThereIsNothingToSay:
    def test_a_config_with_no_video_sources_writes_no_attr(self, tmp_path):
        recorder = _recorder(tmp_path, [])

        RecorderManager._save_camera_serial_data(recorder)

        assert "camera_serial" not in _attrs(recorder)

    def test_a_source_with_no_device_name_is_skipped(self, tmp_path):
        recorder = _recorder(
            tmp_path, [{"name": "nameless", "type": "oak_camera", "params": {}}]
        )

        RecorderManager._save_camera_serial_data(recorder)

        assert "camera_serial" not in _attrs(recorder)

    def test_a_missing_record_path_is_not_an_error(self, tmp_path):
        recorder = _recorder(tmp_path, [])
        recorder.record_path = None

        RecorderManager._save_camera_serial_data(recorder)

    def test_a_missing_episode_group_is_not_an_error(self, tmp_path):
        recorder = _recorder(
            tmp_path, [_source("left_umi_camera", "oak_device_name", "oak-W97-07")]
        )
        recorder.episode_id = 99

        RecorderManager._save_camera_serial_data(recorder)


class TestItSitsBesideTheGripperAttr:
    def test_both_identities_can_coexist_on_one_episode(self, tmp_path):
        """The pair is what makes an episode self-describing: which gripper,
        through which camera."""
        recorder = _recorder(
            tmp_path, [_source("left_umi_camera", "oak_device_name", "oak-W97-07")]
        )
        store = zarr.open(recorder.record_path, mode="a")
        store[f"episode_0{PENDING_SUFFIX}"].attrs["gripper_serial"] = {
            "left_umi_gripper": {"device_name": "UMI-006", "serial": "ABC"}
        }

        RecorderManager._save_camera_serial_data(recorder)

        attrs = _attrs(recorder)
        assert attrs["gripper_serial"]["left_umi_gripper"]["serial"] == "ABC"
        assert attrs["camera_serial"]["left_umi_camera"]["serial"] == OAK_SERIAL


class TestTheWriterIsActuallyCalled:
    """A writer nobody calls is the same as no writer, and this one sits in a
    save path with a dozen other calls where an omission is invisible."""

    def test_the_save_path_invokes_it_beside_the_gripper_writer(self):
        import inspect

        source = inspect.getsource(RecorderManager)
        gripper_at = source.index("self._save_gripper_serial_data()")
        camera_at = source.index("self._save_camera_serial_data()")

        assert camera_at > gripper_at, "camera writer is not in the save path"

    def test_both_writers_are_called_from_the_same_method(self):
        import inspect

        for name, member in inspect.getmembers(RecorderManager, inspect.isfunction):
            body = inspect.getsource(member)
            if "self._save_gripper_serial_data()" in body:
                assert "self._save_camera_serial_data()" in body, (
                    f"{name} saves gripper identity but not camera identity"
                )
                return
        raise AssertionError("no method calls _save_gripper_serial_data")


# Exercise actual template shapes as well as synthetic sources: multi-stream
# OAK identities live at source level, not in params.

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[3] / "config" / "data_recording"

# Every shipped preset. Both declare their wrist OAKs in the streams shape; the
# iPhone rig adds phones under `params`.
TEMPLATES = [
    "production/bimanual_v4.yaml",
    # The same OAK shape with no iphone_camera source at all -- so the writer is
    # exercised on a config where every camera it must find is a stream of an
    # oak_device block, and nothing else is there to carry it.
    "production/bimanual_v4_vive.yaml",
]


def _load(template):
    path = CONFIG_DIR / template
    assert path.exists(), f"{path} is gone -- update the template list"
    return yaml.safe_load(path.read_text())


def _declared_device_names(config):
    """Every camera the config names, and the device behind it."""
    declared = {}
    for source in config.get("video_sources") or []:
        params = source.get("params") or {}
        device = next(
            (
                value
                for key in ("oak_device_name", "iphone_device_name")
                for value in (params.get(key) or source.get(key),)
                if value
            ),
            None,
        )
        if not device:
            continue
        streams = source.get("streams") or []
        names = [s.get("name") for s in streams if s.get("name")] or [
            source.get("name")
        ]
        for name in names:
            if name:
                declared[name] = device
    return declared


def _registry_for(config):
    buckets = {
        "oak_devices": {},
        "iphone_devices": {},
    }
    prefix = {
        "oak": "oak_devices",
        "iph": "iphone_devices",
    }
    for index, device in enumerate(
        sorted(set(_declared_device_names(config).values()))
    ):
        key = prefix.get(device[:3].lower())
        if key:
            buckets[key][device] = f"SERIAL{index:04d}"
    return {
        "version": 1,
        **buckets,
        "umi_v4_grippers": {},
        "gripper_cameras": {},
    }


def _run_against(tmp_path, config, monkeypatch):
    import umi.utils.device_manager as manager
    import umi.utils.device_registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", _registry_for(config), raising=False)
    for name in ("IPHONE_DEVICES", "OAK_DEVICES"):
        getattr(manager, name).invalidate()

    recorder = _recorder(tmp_path, config.get("video_sources") or [])
    RecorderManager._save_camera_serial_data(recorder)
    return _attrs(recorder).get("camera_serial", {})


class TestRealTemplates:
    @pytest.mark.parametrize("template", TEMPLATES)
    def test_every_camera_the_config_declares_is_recorded(
        self, template, tmp_path, monkeypatch
    ):
        config = _load(template)

        stored = _run_against(tmp_path, config, monkeypatch)

        declared = _declared_device_names(config)
        assert set(stored) == set(declared), (
            template,
            sorted(set(declared) - set(stored)),
            sorted(set(stored) - set(declared)),
        )
        for name, device in declared.items():
            assert stored[name]["device_name"] == device, (template, name)

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_no_camera_the_config_does_not_declare_is_invented(
        self, template, tmp_path, monkeypatch
    ):
        """A writer that emitted an entry per <arm>_umi_camera/<arm>_trigger_camera
        pair regardless of the config would pass the test above and fabricate
        entries here -- the trigger cameras are USB modules with no device name
        to record."""
        config = _load(template)

        stored = _run_against(tmp_path, config, monkeypatch)

        assert set(stored) <= set(_declared_device_names(config)), template
        assert all(entry.get("device_name") for entry in stored.values()), template

    @pytest.mark.parametrize("template", ["production/bimanual_v4.yaml"])
    def test_the_per_type_lookup_is_exercised_beyond_oak(
        self, template, tmp_path, monkeypatch
    ):
        config = _load(template)

        stored = _run_against(tmp_path, config, monkeypatch)

        phones = [n for n in stored if n.endswith("_tracking_camera")]
        assert phones, (template, sorted(stored))
        assert all(stored[n].get("serial") for n in phones), template
