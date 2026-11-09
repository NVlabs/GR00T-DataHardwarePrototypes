# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for per-sensor stream resolution in the GUI config editor.

`station_ui.core.device_scan.resolution_for(device_name, socket, scanner)` answers one
question for the config editor: "this OAK, this socket — what output resolution
should the YAML ask for?" The answer is driven by the sensor module the scanner
already probed on that socket, because 640x400 is 16:10: that matches the native
aspect of AR0234 / OV9782 / OV9282, but *not* IMX378 (4:3). Asking a 4:3 module
for a 16:10 output makes the ISP centre-crop and silently throw away FOV. So the
table pins width 640 at each module's own native aspect.

Two classes of behaviour are pinned here:

* Fallback hygiene. Every miss — unknown sensor, device not in the scan, socket
  not on that device, malformed metadata — must return DEFAULT_STREAM_RESOLUTION
  *and* a non-empty warning. A silent default is the failure mode that matters:
  it looks exactly like a correct answer and quietly costs FOV.
* The FALLBACK_INTRINSICS invariant. `oak_device.fallback_intrinsics()` is keyed
  by (sensorName, (width, height)). If this table hands out a resolution that
  table is not keyed at, the lookup misses, returns None, and the camera runs
  with no intrinsics at all — no exception anywhere. See
  `test_every_shared_sensor_is_keyed_at_the_same_resolution`.

Fully hardware-free: the scanner's probe callables are injected fakes, so no OAK
needs to be attached and depthai is never opened.

Run from project root:
    uv run pytest tests/gui/test_oak_resolution.py -v
"""

import copy
import sys
from pathlib import Path

import pytest

# Project root so that "umi" and "station_ui" resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from station_ui.core.device_scan import (  # noqa: E402
    DEFAULT_STREAM_RESOLUTION,
    SENSOR_STREAM_RESOLUTION,
    DeviceScanner,
    resolution_for,
)
from umi.common.hardware_sdk.device_discovery import (  # noqa: E402
    ConnectedDevice,
)

# oak_device imports depthai at module import time; keep the rest of this file
# collectable on machines without it.
try:
    from umi.common.hardware_sdk.camera.oak_device import (
        FALLBACK_INTRINSICS,
        fallback_intrinsics,
    )
except Exception:  # pragma: no cover - depends on the local install
    FALLBACK_INTRINSICS = None
    fallback_intrinsics = None

# Snapshotted at import, before any test can mutate the real tables. Expected
# values are compared against these copies so that a leak in one test cannot
# quietly rewrite what the other tests consider correct.
_TABLE_SNAPSHOT = copy.deepcopy(SENSOR_STREAM_RESOLUTION)
_DEFAULT_SNAPSHOT = list(DEFAULT_STREAM_RESOLUTION)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _oak(registry_name, sockets, state="available", device_id=None):
    """An OAK entry shaped exactly like list_connected_oak() produces."""
    return ConnectedDevice(
        kind="oak",
        device_id=device_id or f"oak-id-{registry_name}",
        registry_name=registry_name,
        state=state,
        metadata={"sockets": dict(sockets)},
    )


def _oak_raw_metadata(registry_name, metadata, device_id="oak-id-raw"):
    """An OAK entry with hand-built metadata, for the malformed-payload cases."""
    return ConnectedDevice(
        kind="oak",
        device_id=device_id,
        registry_name=registry_name,
        state="available",
        metadata=metadata,
    )


def _unscanned_scanner(*devices):
    """A DeviceScanner wired to fakes only — no probe touches hardware."""
    return DeviceScanner(
        scan_oak=lambda: list(devices),
        scan_iphone=lambda: [],
        scan_gripper=lambda: [],
    )


def _scanner_with(*devices):
    """A scanned DeviceScanner holding exactly `devices` as its OAK results."""
    scanner = _unscanned_scanner(*devices)
    scanner.scan()
    return scanner


def _assert_shape(resolution):
    assert isinstance(resolution, list), f"expected a list, got {type(resolution)}"
    assert len(resolution) == 2, f"expected [width, height], got {resolution!r}"
    for value in resolution:
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value > 0


def _assert_warned(warning, *needles):
    """A miss must produce a human-readable warning, not a bare None."""
    assert warning is not None, "a fallback must be reported, not silent"
    assert isinstance(warning, str), f"warning must be a string, got {type(warning)}"
    assert warning.strip(), "warning must not be empty"
    for needle in needles:
        assert needle in warning, (
            f"warning should name {needle!r} so the operator can act on it; got: "
            f"{warning!r}"
        )


def _assert_fell_back(result, context):
    """A miss must be the default resolution AND say something about it."""
    assert isinstance(result, tuple) and len(result) == 2, (
        f"resolution_for must return (resolution, warning); got {result!r}"
    )
    resolution, warning = result
    _assert_shape(resolution)
    assert resolution == _DEFAULT_SNAPSHOT, (
        f"{context}: expected the default {_DEFAULT_SNAPSHOT}, got {resolution}"
    )
    assert isinstance(warning, str) and warning.strip(), (
        f"{context}: fell back to the default with no warning ({warning!r}). "
        "A silent default is indistinguishable from a correct answer, so a "
        "mis-typed device name or a swapped module costs FOV with nothing in "
        "the log to explain it."
    )


# --------------------------------------------------------------------------
# 1. known module -> table value
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sensor, expected",
    [
        ("AR0234", [640, 400]),  # 1920x1200 native, 16:10
        ("OV9782", [640, 400]),  # 1280x800 native, 16:10
        ("OV9282", [640, 400]),  # 1280x800 native, 16:10
        ("IMX378", [640, 480]),  # 4056x3040 native, 4:3
    ],
)
def test_known_module_gets_its_native_aspect(sensor, expected):
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": sensor}))

    resolution, warning = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    assert resolution == expected
    assert warning is None, f"a known module must not warn; got {warning!r}"
    _assert_shape(resolution)


def test_imx378_is_four_three_not_sixteen_ten():
    """The whole point: a 4:3 module must not be asked for a 16:10 output.

    640x400 on an IMX378 makes the ISP centre-crop to 16:10 and silently
    discard vertical FOV.
    """
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "IMX378"}))

    resolution, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    width, height = resolution
    assert (width, height) == (640, 480)
    assert width / height == pytest.approx(4 / 3)


def test_ar0234_is_sixteen_ten():
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "AR0234"}))

    resolution, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    width, height = resolution
    assert (width, height) == (640, 400)
    assert width / height == pytest.approx(16 / 10)


def test_every_table_entry_is_reachable_through_resolution_for():
    """Whatever the table says today, the lookup must actually serve it."""
    for sensor, expected in _TABLE_SNAPSHOT.items():
        scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": sensor}))
        resolution, warning = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)
        assert resolution == list(expected), f"wrong resolution for {sensor}"
        assert warning is None, f"{sensor} is in the table and must not warn"


# --------------------------------------------------------------------------
# 2. unknown sensorName -> default + warning that names the sensor
# --------------------------------------------------------------------------


def test_unknown_sensor_falls_back_to_default_with_a_warning():
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "IMX999"}))

    result = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    _assert_fell_back(result, "unknown sensor IMX999")
    _assert_warned(result[1], "IMX999")


def test_unknown_sensor_warning_names_the_sensor():
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "IMX999"}))

    _, warning = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    assert "IMX999" in warning, (
        "the warning must name the unrecognised module, otherwise nobody can "
        f"tell which entry SENSOR_STREAM_RESOLUTION is missing; got {warning!r}"
    )


@pytest.mark.parametrize("sensor", [None, "", "   "])
def test_empty_sensor_name_falls_back(sensor):
    """A socket that reported no module at all is still just a miss."""
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": sensor}))

    result = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    _assert_fell_back(result, f"socket reporting sensorName={sensor!r}")


def test_lowercase_sensor_name_is_never_silently_wrong():
    """depthai reports upper-case names; a lower-case one is either normalised
    or a miss, but must never come back as a quiet 16:10 default for a 4:3
    module."""
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "imx378"}))

    resolution, warning = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    _assert_shape(resolution)
    if warning is None:
        assert resolution == _TABLE_SNAPSHOT["IMX378"], (
            "no warning means the lookup claimed a hit, so it must be the "
            f"IMX378 row; got {resolution}"
        )
    else:
        assert resolution == _DEFAULT_SNAPSHOT


# --------------------------------------------------------------------------
# 3. device absent from the scan -> default + warning
# --------------------------------------------------------------------------


def test_device_absent_from_scan_falls_back():
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "AR0234"}))

    result = resolution_for("oak-FFC-99", "CAM_A", scanner=scanner)

    _assert_fell_back(result, "device oak-FFC-99 not in the scan")


def test_empty_scan_falls_back():
    """Config editing offline (nothing plugged in) must still work."""
    result = resolution_for("oak-FFC-01", "CAM_A", scanner=_scanner_with())

    _assert_fell_back(result, "empty scan")


def test_never_scanned_scanner_falls_back():
    """A scanner that has not run yet holds no devices."""
    scanner = _unscanned_scanner(_oak("oak-FFC-01", {"CAM_A": "IMX378"}))

    result = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    _assert_fell_back(result, "scanner never scanned")


@pytest.mark.parametrize("device_name", [None, "", "   "])
def test_empty_device_name_falls_back(device_name):
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "IMX378"}))

    result = resolution_for(device_name, "CAM_A", scanner=scanner)

    _assert_fell_back(result, f"device_name={device_name!r}")


def test_none_device_name_does_not_match_an_unregistered_device():
    """Unregistered devices carry registry_name=None; a None lookup must not
    collide with them and inherit their modules."""
    unregistered = _oak(None, {"CAM_A": "IMX378"}, device_id="oak-id-unknown")

    result = resolution_for(None, "CAM_A", scanner=_scanner_with(unregistered))

    _assert_fell_back(result, "device_name=None vs unregistered device")


def test_device_id_resolves_the_same_as_a_registry_name():
    """A config may identify a device by id.

    Unregistered hardware has no registry name, so the form stores its id and
    the config writes oak_device_id. Refusing ids here would make such a device
    always fall back to the default resolution -- an IMX378 would get 640x400
    and lose FOV to a centre crop.
    """
    device = _oak("oak-FFC-01", {"CAM_A": "IMX378"}, device_id="14442C10D1")

    resolution, warning = resolution_for(
        "14442C10D1", "CAM_A", scanner=_scanner_with(device)
    )

    assert resolution == [640, 480]
    assert warning is None


def test_scanner_none_uses_the_module_scanner_and_still_falls_back(monkeypatch):
    """Omitting `scanner` must not raise, even with nothing ever scanned."""
    monkeypatch.setattr(
        "station_ui.core.device_scan.SCANNER", _scanner_with(), raising=True
    )

    result = resolution_for("no-such-device-anywhere", "CAM_A")

    _assert_fell_back(result, "default module-level scanner")


# --------------------------------------------------------------------------
# 4. socket missing from that device -> default + warning
# --------------------------------------------------------------------------


def test_socket_missing_from_device_falls_back():
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "AR0234"}))

    result = resolution_for("oak-FFC-01", "CAM_D", scanner=scanner)

    _assert_fell_back(result, "socket CAM_D not on oak-FFC-01")


@pytest.mark.parametrize("socket", [None, "", "   "])
def test_empty_socket_falls_back(socket):
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "AR0234"}))

    result = resolution_for("oak-FFC-01", socket, scanner=scanner)

    _assert_fell_back(result, f"socket={socket!r}")


def test_lowercase_socket_is_never_silently_wrong():
    """Sockets are reported as CAM_A/CAM_B/...; a lower-case spelling either
    normalises to the right module or is a loud miss."""
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "IMX378"}))

    resolution, warning = resolution_for("oak-FFC-01", "cam_a", scanner=scanner)

    _assert_shape(resolution)
    if warning is None:
        assert resolution == _TABLE_SNAPSHOT["IMX378"], (
            "no warning means the socket was resolved, so it must be the "
            f"IMX378 row; got {resolution}"
        )
    else:
        assert resolution == _DEFAULT_SNAPSHOT


@pytest.mark.parametrize(
    "metadata",
    [
        {},  # no sockets key at all (e.g. an iPhone entry)
        {"sockets": {}},  # probed but nothing reported
        {"sockets": None},  # probe wrote a null payload
    ],
    ids=["no-sockets-key", "empty-sockets", "null-sockets"],
)
def test_malformed_socket_metadata_falls_back(metadata):
    scanner = _scanner_with(_oak_raw_metadata("oak-FFC-01", metadata))

    result = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    _assert_fell_back(result, f"metadata={metadata!r}")


def test_unavailable_device_falls_back():
    """A device that enumerated but would not open reports no sockets."""
    device = _oak("oak-FFC-01", {}, state="unavailable")

    result = resolution_for("oak-FFC-01", "CAM_A", scanner=_scanner_with(device))

    _assert_fell_back(result, "unavailable device")


# --------------------------------------------------------------------------
# 5. returned lists are copies
# --------------------------------------------------------------------------


def test_mutating_a_hit_result_does_not_corrupt_the_table():
    """Handing out the table's own list lets one caller resize every camera."""
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "IMX378"}))

    first, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)
    first[0] = -1
    first.append(9999)

    second, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)
    assert second == _TABLE_SNAPSHOT["IMX378"], (
        "a caller that edited its own resolution list poisoned the next lookup"
    )
    assert SENSOR_STREAM_RESOLUTION == _TABLE_SNAPSHOT, (
        "SENSOR_STREAM_RESOLUTION was mutated through a returned list; the "
        "table must be handed out as copies, otherwise editing one config "
        "field rewrites the shared sensor table for every later camera"
    )


def test_hit_result_is_not_the_table_entry_itself():
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "AR0234"}))

    resolution, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    assert resolution is not SENSOR_STREAM_RESOLUTION["AR0234"]


def test_each_call_returns_an_independent_list():
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "AR0234"}))

    first, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)
    second, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    assert first == second
    assert first is not second, "two calls must not share one list object"
    first.append(999)
    assert second == _TABLE_SNAPSHOT["AR0234"]


def test_mutating_a_fallback_result_does_not_corrupt_the_default():
    scanner = _scanner_with()

    first, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)
    first[0] = -1
    first.append(9999)

    second, _ = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)
    assert second == _DEFAULT_SNAPSHOT
    assert DEFAULT_STREAM_RESOLUTION == _DEFAULT_SNAPSHOT, (
        "DEFAULT_STREAM_RESOLUTION was mutated through a returned list; one "
        "mutation changes the fallback for every later camera"
    )


def test_fallback_result_is_not_the_default_constant_itself():
    resolution, _ = resolution_for("nope", "CAM_A", scanner=_scanner_with())

    assert resolution is not DEFAULT_STREAM_RESOLUTION


# --------------------------------------------------------------------------
# 6. mixed modules on one device
# --------------------------------------------------------------------------


def test_one_device_two_modules_two_resolutions():
    """A head OAK carries a 4:3 IMX378 and a 16:10 OV9782 at once.

    One resolution for the whole device would be wrong for one of them.
    """
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "IMX378", "CAM_B": "OV9782"}))

    cam_a, warn_a = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)
    cam_b, warn_b = resolution_for("oak-FFC-01", "CAM_B", scanner=scanner)

    assert cam_a == [640, 480], "IMX378 is 4:3; 16:10 would centre-crop the FOV"
    assert cam_b == [640, 400], "OV9782 is 16:10"
    assert cam_a != cam_b, "per-socket lookup collapsed to a single resolution"
    assert warn_a is None and warn_b is None


def test_three_sockets_resolve_independently():
    scanner = _scanner_with(
        _oak(
            "oak-FFC-01",
            {"CAM_A": "IMX378", "CAM_B": "OV9282", "CAM_C": "AR0234"},
        )
    )

    resolved = {
        socket: resolution_for("oak-FFC-01", socket, scanner=scanner)[0]
        for socket in ("CAM_A", "CAM_B", "CAM_C")
    }

    assert resolved == {
        "CAM_A": [640, 480],
        "CAM_B": [640, 400],
        "CAM_C": [640, 400],
    }


def test_resolution_is_per_device_not_first_match():
    """Two devices, same socket name, different modules."""
    scanner = _scanner_with(
        _oak("oak-FFC-01", {"CAM_A": "AR0234"}),
        _oak("oak-FFC-02", {"CAM_A": "IMX378"}),
    )

    assert resolution_for("oak-FFC-02", "CAM_A", scanner=scanner)[0] == [640, 480]
    assert resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)[0] == [640, 400]


def test_one_bad_socket_does_not_poison_its_neighbour():
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": "IMX999", "CAM_B": "IMX378"}))

    bad = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)
    _assert_fell_back(bad, "unknown IMX999")
    _assert_warned(bad[1], "IMX999")

    cam_b, warn_b = resolution_for("oak-FFC-01", "CAM_B", scanner=scanner)
    assert cam_b == [640, 480]
    assert warn_b is None


# --------------------------------------------------------------------------
# 7. Invariant: shared sensors must be keyed at the same resolution
# --------------------------------------------------------------------------

_FALLBACK_RESOLUTIONS_BY_SENSOR = {}
if FALLBACK_INTRINSICS is not None:
    for _sensor, _res in FALLBACK_INTRINSICS:
        _FALLBACK_RESOLUTIONS_BY_SENSOR.setdefault(_sensor, set()).add(tuple(_res))

_SHARED_SENSORS = sorted(set(_TABLE_SNAPSHOT) & set(_FALLBACK_RESOLUTIONS_BY_SENSOR))

_NO_SHARED_SENSORS = [
    pytest.param(
        None,
        marks=pytest.mark.skip(
            reason="no sensor appears in both FALLBACK_INTRINSICS and "
            "SENSOR_STREAM_RESOLUTION (or depthai is not installed, so "
            "oak_device could not be imported)"
        ),
    )
]

_INVARIANT_EXPLANATION = (
    "oak_device.fallback_intrinsics() is a dict lookup keyed by "
    "(sensorName, (width, height)). It is the only source of intrinsics for "
    "modules with no on-board factory calibration. If SENSOR_STREAM_RESOLUTION "
    "streams that sensor at a resolution FALLBACK_INTRINSICS is not keyed at, "
    "the lookup misses, returns None, and the camera runs with NO intrinsics at "
    "all -- no exception, no warning, just a silent calibration hole "
    "downstream, discovered much later as unusable data. Fix by keying both "
    "tables at the same resolution: either add the new (sensor, resolution) "
    "entry to FALLBACK_INTRINSICS, or leave SENSOR_STREAM_RESOLUTION at the "
    "resolution the intrinsics were measured at."
)


@pytest.mark.parametrize("sensor", _SHARED_SENSORS or _NO_SHARED_SENSORS)
def test_every_shared_sensor_is_keyed_at_the_same_resolution(sensor):
    stream = tuple(SENSOR_STREAM_RESOLUTION[sensor])
    keyed = _FALLBACK_RESOLUTIONS_BY_SENSOR[sensor]

    assert stream in keyed, (
        f"{sensor}: SENSOR_STREAM_RESOLUTION streams at {stream}, but "
        f"FALLBACK_INTRINSICS only has {sorted(keyed)}.\n" + _INVARIANT_EXPLANATION
    )


@pytest.mark.parametrize("sensor", _SHARED_SENSORS or _NO_SHARED_SENSORS)
def test_hardcoded_intrinsics_survive_a_real_resolution_lookup(sensor):
    """End-to-end version of the invariant: scan a device carrying this module,
    take the resolution the GUI would write, and ask for its fallback K."""
    scanner = _scanner_with(_oak("oak-FFC-01", {"CAM_A": sensor}))

    resolution, warning = resolution_for("oak-FFC-01", "CAM_A", scanner=scanner)

    assert warning is None
    K = fallback_intrinsics(sensor, tuple(resolution))
    assert K is not None, (
        f"{sensor}: resolution_for() returned {resolution}, and "
        f"fallback_intrinsics({sensor!r}, {tuple(resolution)}) found nothing.\n"
        + _INVARIANT_EXPLANATION
    )


# --------------------------------------------------------------------------
# table sanity
# --------------------------------------------------------------------------


def test_table_and_default_match_the_documented_contract():
    assert DEFAULT_STREAM_RESOLUTION == [640, 400]
    for sensor, expected in (
        ("AR0234", [640, 400]),
        ("OV9782", [640, 400]),
        ("OV9282", [640, 400]),
        ("IMX378", [640, 480]),
    ):
        assert sensor in SENSOR_STREAM_RESOLUTION, f"{sensor} missing from table"
        assert list(SENSOR_STREAM_RESOLUTION[sensor]) == expected


@pytest.mark.parametrize("sensor", sorted(_TABLE_SNAPSHOT))
def test_every_table_entry_is_a_two_element_int_list(sensor):
    _assert_shape(SENSOR_STREAM_RESOLUTION[sensor])


@pytest.mark.parametrize("sensor", sorted(_TABLE_SNAPSHOT))
def test_every_table_entry_is_640_wide(sensor):
    """The table varies height to follow native aspect; width stays 640."""
    assert SENSOR_STREAM_RESOLUTION[sensor][0] == 640


def test_imx378_is_the_odd_one_out():
    """The whole point of the table: the 4:3 module must not get 16:10."""
    assert SENSOR_STREAM_RESOLUTION["IMX378"] != _DEFAULT_SNAPSHOT
    width, height = SENSOR_STREAM_RESOLUTION["IMX378"]
    assert width * 3 == height * 4, "IMX378 is 4:3 native"
    for sensor in ("AR0234", "OV9782", "OV9282"):
        width, height = SENSOR_STREAM_RESOLUTION[sensor]
        assert width * 10 == height * 16, f"{sensor} is 16:10 native"
