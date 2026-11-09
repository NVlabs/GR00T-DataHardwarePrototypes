# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-camera coverage for the marker device-assignment gate."""

import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("depthai")
pytest.importorskip("record3d")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from real_scripts.device_verification import (  # noqa: E402
    verify_device_assignment as verify,
)
from station_ui import markers  # noqa: E402
from station_ui.calibration import payload_for_side  # noqa: E402
from station_ui.calibration import (  # noqa: E402
    verification_roles as _device_verification_roles,
)


def _write_config(tmp_path, *, use_head_mount=True, sources=()):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {"use_head_mount": use_head_mount},
                "video_sources": list(sources),
            }
        )
    )
    return path


@pytest.mark.parametrize(
    ("identifier_key", "identifier"),
    [
        ("oak_device_name", "oak-W97-01"),
        ("oak_device_id", "1944301099AABB5A00"),
    ],
)
def test_head_umi_role_resolves_registered_names_and_raw_ids(
    tmp_path, identifier_key, identifier
):
    path = _write_config(
        tmp_path,
        sources=[
            {
                "type": "oak_camera",
                "name": "head_umi_camera",
                "params": {identifier_key: identifier},
            }
        ],
    )

    camera_type, kwargs = verify.resolve_camera_for_role(path, "head_umi")

    assert camera_type == "oak"
    assert kwargs[identifier_key] == identifier
    sibling = (
        "oak_device_id" if identifier_key == "oak_device_name" else "oak_device_name"
    )
    assert kwargs[sibling] is None
    assert verify.payload_for_role("head_umi") == "UMI_VERIFY_HEAD"


def test_both_head_cameras_are_verified_independently():
    config = {
        "meta": {"use_head_mount": True},
        "video_sources": [
            {"type": "iphone_camera", "name": "head_tracking_camera"},
            {"type": "oak_camera", "name": "head_umi_camera"},
        ],
    }

    assert _device_verification_roles(config) == ["head", "head_umi"]


def test_disabled_head_mount_schedules_and_resolves_no_head_camera(tmp_path):
    sources = [
        {"type": "iphone_camera", "name": "head_tracking_camera"},
        {"type": "oak_camera", "name": "head_umi_camera"},
    ]
    config = {"meta": {"use_head_mount": False}, "video_sources": sources}
    path = _write_config(tmp_path, use_head_mount=False, sources=sources)

    assert _device_verification_roles(config) == []
    assert verify.resolve_camera_for_role(path, "head") == (None, None)
    assert verify.resolve_camera_for_role(path, "head_umi") == (None, None)


def test_head_roles_resolve_the_generated_descriptor_names(tmp_path):
    """The GUI names head cameras from the optional-device descriptors.
    Recognising only older names would make every generated config skip head
    verification without saying so."""
    path = _write_config(
        tmp_path,
        sources=[
            {
                "type": "iphone_camera",
                "name": "head_iphone_camera",
                "params": {"iphone_device_name": "iphone16p-ry"},
            },
            {
                "type": "oak_camera",
                "name": "head_oak_camera",
                "params": {"oak_device_name": "oak-W97-51"},
            },
        ],
    )

    camera_type, kwargs = verify.resolve_camera_for_role(path, "head")
    assert camera_type == "iphone"
    assert kwargs["iphone_device_name"] == "iphone16p-ry"

    camera_type, kwargs = verify.resolve_camera_for_role(path, "head_umi")
    assert camera_type == "oak"
    assert kwargs["oak_device_name"] == "oak-W97-51"


@pytest.mark.parametrize(
    ("role", "name", "expected_type"),
    [
        ("head", "head_tracking_camera", "iphone"),
        ("head", "head_mount_iphone", "iphone"),
        ("head_umi", "head_umi_camera", "oak"),
    ],
)
def test_legacy_head_names_still_resolve(tmp_path, role, name, expected_type):
    source_type = "iphone_camera" if expected_type == "iphone" else "oak_camera"
    path = _write_config(
        tmp_path, sources=[{"type": source_type, "name": name, "params": {}}]
    )

    camera_type, _ = verify.resolve_camera_for_role(path, role)

    assert camera_type == expected_type


def test_an_unresolved_role_says_so_instead_of_skipping_silently(tmp_path, capsys):
    path = _write_config(tmp_path, sources=[])

    assert verify.resolve_camera_for_role(path, "left_tracking") == (None, None)

    err = capsys.readouterr().err
    assert "Left tracking" in err
    assert "left_tracking_camera" in err


def test_a_head_role_says_why_it_was_skipped(tmp_path, capsys):
    sources = [{"type": "iphone_camera", "name": "head_iphone_camera"}]
    path = _write_config(tmp_path, use_head_mount=False, sources=sources)

    assert verify.resolve_camera_for_role(path, "head") == (None, None)

    assert "use_head_mount" in capsys.readouterr().err


def test_raw_identifiers_reach_camera_constructors(monkeypatch):
    captured = {}

    class _Camera:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(verify, "OAKCamera", _Camera)
    verify.make_verify_camera(
        "oak",
        {
            "oak_device_name": None,
            "oak_device_id": "1944301099AABB5A00",
        },
    )

    assert captured == {
        "oak_device_name": None,
        "oak_device_id": "1944301099AABB5A00",
    }


def test_the_bottom_oak_is_never_opened_by_the_check(tmp_path):
    """The static table OAK is recorded but not checked. `--watch` plans its
    devices from every role it knows, so a config whose only camera is the
    bottom OAK plans nothing: the check neither opens it nor waits on it."""
    sources = [
        {
            "type": "oak_camera",
            "name": "bottom_oak_camera",
            "params": {"oak_device_name": "oak-W97-77"},
        }
    ]
    path = _write_config(tmp_path, use_head_mount=False, sources=sources)

    assert verify.plan_devices(path, list(verify.ROLE_LABELS)) == []
    assert (
        _device_verification_roles(
            {"meta": {"use_head_mount": False}, "video_sources": sources}
        )
        == []
    )


# ------------------------------------------------------- the marker round trip


def test_every_marker_the_page_draws_is_one_the_worker_decodes():
    """The page renders `station_ui.markers.MARKER_IDS`; the worker translates
    a detected id back into that side's payload. The worker's table is derived
    from the page's, so a new side cannot be drawn and detected but then dropped
    for a missing lookup -- which would leave its camera in WAITING however the
    operator aimed, with a healthy preview. This pins that they stay in step."""
    assert verify._ARUCO_SIDE_BY_ID == {
        marker_id: side.upper() for side, marker_id in markers.MARKER_IDS.items()
    }


def _rasterised(side: str, scale: int = 20) -> np.ndarray:
    """The marker the page serves for *side*, as a camera would see it: white
    field, black modules, one module of quiet zone, in BGR."""
    grid = markers.marker_matrix(side)
    quiet = 1
    span = len(grid) + 2 * quiet
    image = np.full((span * scale, span * scale), 255, np.uint8)
    for y, row in enumerate(grid):
        for x, module in enumerate(row):
            if module == "1":
                top, left = (y + quiet) * scale, (x + quiet) * scale
                image[top : top + scale, left : left + scale] = 0
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("side", sorted(markers.MARKER_IDS))
def test_the_rendered_marker_decodes_to_its_sides_payload(side):
    """End to end through the worker's own decoder: what the page serves for
    *side* comes back as the payload `station_ui.calibration.fold_events`
    compares against. Every side, so a fifth marker cannot ship undecodable."""
    if not hasattr(cv2, "aruco"):
        pytest.skip("this OpenCV build has no aruco module")

    assert verify.decode_marker(_rasterised(side)) == payload_for_side(side)
