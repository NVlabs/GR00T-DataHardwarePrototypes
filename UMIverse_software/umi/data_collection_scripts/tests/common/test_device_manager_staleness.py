# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `_LazyDict` singletons must follow the registry, not latch on it.

`device_manager` exposes eight module-level dicts that load from the registry on
first access. If they did not notice `device_registry._REGISTRY` being replaced,
they would serve their first snapshot forever.

Two consequences, and the second is the one that matters more:

  - Test ordering. A test file that populates the maps from its own fixture
    would change the results of whichever file runs after it.
  - Production. `device_registry.reload()` replaces `_REGISTRY` by design --
    that is its whole purpose. A long-lived process that reloaded would keep
    serving the maps it had loaded first, silently, with no error and no log.

So the dicts notice the change themselves rather than relying on a fixture to
reset them: a fixture would fix the tests and leave the station wrong.
"""

import json

import pytest

import umi.utils.device_manager as manager
import umi.utils.device_registry as registry


def _snapshot(oak):
    return {
        "version": 1,
        "iphone_devices": {},
        "oak_devices": dict(oak),
        "umi_v4_grippers": {},
        "gripper_cameras": {},
        "device_metadata": {},
    }


@pytest.fixture(autouse=True)
def _clean_registry():
    registry._REGISTRY = None
    yield
    registry._REGISTRY = None


class TestTheMapsFollowTheRegistry:
    def test_replacing_the_registry_is_noticed(self):
        """Test ordering, reduced: no manual reset in between."""
        registry._REGISTRY = _snapshot({"oak-FFC-01": "S1"})
        assert dict(manager.OAK_DEVICES) == {"oak-FFC-01": "S1"}

        registry._REGISTRY = _snapshot({"oak-FFC-99": "S99"})

        assert dict(manager.OAK_DEVICES) == {"oak-FFC-99": "S99"}

    def test_a_reload_is_reflected(self, monkeypatch, tmp_path):
        """The production half. `reload()` exists to replace the snapshot, and
        a station that reloads must not keep serving the maps it loaded first."""
        registry._REGISTRY = _snapshot({"oak-FFC-01": "S1"})
        assert "oak-FFC-01" in manager.OAK_DEVICES

        path = tmp_path / "devices.json"
        path.write_text(json.dumps(_snapshot({"oak-FFC-07": "S7"})))
        monkeypatch.setenv(registry.ENV_REGISTRY_FILE, str(path))
        registry.reload()

        assert dict(manager.OAK_DEVICES) == {"oak-FFC-07": "S7"}

    def test_a_removed_device_disappears(self):
        """`update()` alone would leave the old entry behind -- the map has to
        be cleared, not merged."""
        registry._REGISTRY = _snapshot({"oak-FFC-01": "S1", "oak-FFC-02": "S2"})
        assert len(manager.OAK_DEVICES) == 2

        registry._REGISTRY = _snapshot({"oak-FFC-01": "S1"})

        assert dict(manager.OAK_DEVICES) == {"oak-FFC-01": "S1"}

    def test_case_insensitive_lookup_still_works_after_a_change(self):
        registry._REGISTRY = _snapshot({"oak-FFC-01": "S1"})
        assert manager.OAK_DEVICES["OAK-ffc-01"] == "S1"

        registry._REGISTRY = _snapshot({"oak-FFC-99": "S99"})

        assert manager.OAK_DEVICES["oak-ffc-99"] == "S99"


class TestLazinessIsPreserved:
    """The fix must not turn these into eager or per-access loads. They exist
    to avoid reading the registry at import, and re-running the loader on every lookup
    would rebuild eight dicts per call."""

    def test_the_loader_does_not_run_again_while_the_registry_is_unchanged(self):
        registry._REGISTRY = _snapshot({"oak-FFC-01": "S1"})
        calls = {"n": 0}

        def _counting_loader():
            calls["n"] += 1
            return registry.get_bucket("oak_devices")

        lazy = manager._LazyDict(_counting_loader)

        for _ in range(5):
            _ = dict(lazy)
            _ = lazy.get("oak-FFC-01")

        assert calls["n"] == 1, calls

    def test_the_loader_runs_again_exactly_once_per_change(self):
        registry._REGISTRY = _snapshot({"oak-FFC-01": "S1"})
        calls = {"n": 0}

        def _counting_loader():
            calls["n"] += 1
            return registry.get_bucket("oak_devices")

        lazy = manager._LazyDict(_counting_loader)
        _ = dict(lazy)
        registry._REGISTRY = _snapshot({"oak-FFC-02": "S2"})
        _ = dict(lazy)
        _ = dict(lazy)

        assert calls["n"] == 2, calls

    def test_nothing_loads_before_first_access(self):
        calls = {"n": 0}

        def _counting_loader():
            calls["n"] += 1
            return {}

        manager._LazyDict(_counting_loader)

        assert calls["n"] == 0
