# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for umi.common.hardware_sdk.camera.intrinsics_calibration.

Everything here is offline and deterministic: the calibration round-trips are
driven by synthetically projected board corners (no camera, no image files),
and the detector tests render the ChArUco board with OpenCV itself.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from umi.common.hardware_sdk.camera.intrinsics_calibration import (
    ARUCO_DICTS,
    CalibrationResult,
    CharucoBoardSpec,
    Detection,
    calibrate_intrinsics,
    coverage_fraction,
    coverage_grid,
    detect_charuco,
    sharpness_score,
)

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------

SQUARES_X = 14
SQUARES_Y = 9
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.03
ARUCO_DICT_NAME = "DICT_5X5_1000"
NUM_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)  # 104

IMAGE_W, IMAGE_H = 640, 400
IMAGE_SIZE = (IMAGE_W, IMAGE_H)  # (width, height)
IMAGE_SHAPE = (IMAGE_H, IMAGE_W)  # (height, width)

FISHEYE_K_GT = np.array(
    [[310.0, 0.0, 320.0], [0.0, 310.0, 200.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
FISHEYE_D_GT = np.array([-0.03, 0.005, -0.001, 0.0], dtype=np.float64)

PINHOLE_K_GT = np.array(
    [[520.0, 0.0, 318.0], [0.0, 515.0, 205.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
# rational model: k1 k2 p1 p2 k3 k4 k5 k6
PINHOLE_D_GT = np.array(
    [-0.28, 0.09, 0.001, -0.0005, 0.01, 0.02, -0.005, 0.002], dtype=np.float64
)

# 15 viewing directions (azimuth, elevation in degrees) around the board
# normal. Deliberately not fronto-parallel-only: without tilt the focal length
# is not observable from a planar target.
VIEW_ANGLES_DEG = [
    (0.0, 0.0),
    (26.0, 0.0),
    (-26.0, 0.0),
    (0.0, 24.0),
    (0.0, -24.0),
    (20.0, 18.0),
    (-20.0, 18.0),
    (20.0, -18.0),
    (-20.0, -18.0),
    (32.0, 10.0),
    (-32.0, -10.0),
    (12.0, 30.0),
    (-12.0, -30.0),
    (28.0, 24.0),
    (-28.0, 24.0),
]

# Allow 1% relative intrinsic error and 0.1 pixel RMS for synthetic calibration
# round-trip tolerance.
REL_TOL = 0.01
RMS_TOL = 0.1


# --------------------------------------------------------------------------
# Synthetic-projection helpers
# --------------------------------------------------------------------------


def _look_at(cam_pos, target, up):
    """World->camera rotation + translation for a camera at `cam_pos`."""
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    return R, -R @ cam_pos


def _camera_poses(board_center, base_distance, distance_span):
    """(rvec, tvec) for each entry in VIEW_ANGLES_DEG, varied in tilt and roll."""
    poses = []
    for i, (az_deg, el_deg) in enumerate(VIEW_ANGLES_DEG):
        az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
        distance = base_distance + distance_span * ((i % 4) / 3.0)
        cam_pos = board_center + distance * np.array(
            [np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)]
        )
        # vary roll too, so the pose set is not degenerate in image rotation
        roll = np.deg2rad(((i * 37) % 90) - 45)
        up = np.array([np.sin(roll), -np.cos(roll), 0.0])
        R, t = _look_at(cam_pos, board_center, up)
        rvec, _ = cv2.Rodrigues(R)
        poses.append((rvec, t.reshape(3, 1)))
    return poses


def _make_detection(points_xy, ids, source=""):
    """Build a Detection with exactly the dtypes/shapes the contract mandates."""
    corners = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
    return Detection(
        charuco_corners=corners,
        charuco_ids=ids,
        image_shape=IMAGE_SHAPE,
        source=source,
    )


def _synthetic_detections(board, model):
    """Project the board's ground-truth 3D corners through a known K/D."""
    obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    center = obj.mean(axis=0)
    all_ids = np.arange(obj.shape[0], dtype=np.int32)

    if model == "fisheye":
        poses = _camera_poses(center, base_distance=0.42, distance_span=0.10)
        K_gt, D_gt = FISHEYE_K_GT, FISHEYE_D_GT
    else:
        poses = _camera_poses(center, base_distance=0.70, distance_span=0.18)
        K_gt, D_gt = PINHOLE_K_GT, PINHOLE_D_GT

    detections = []
    for i, (rvec, tvec) in enumerate(poses):
        if model == "fisheye":
            pts, _ = cv2.fisheye.projectPoints(
                obj.reshape(-1, 1, 3), rvec, tvec, K_gt, D_gt
            )
        else:
            pts, _ = cv2.projectPoints(obj, rvec, tvec, K_gt, D_gt)
        detections.append(
            _make_detection(pts.reshape(-1, 2), all_ids, source=f"view_{i:02d}")
        )
    return detections


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def board_spec():
    return CharucoBoardSpec(
        squares_x=SQUARES_X,
        squares_y=SQUARES_Y,
        square_length=SQUARE_LENGTH,
        marker_length=MARKER_LENGTH,
        aruco_dict=ARUCO_DICT_NAME,
    )


@pytest.fixture(scope="module")
def board(board_spec):
    return board_spec.build()


@pytest.fixture(scope="module")
def fisheye_detections(board):
    return _synthetic_detections(board, "fisheye")


@pytest.fixture(scope="module")
def pinhole_detections(board):
    return _synthetic_detections(board, "pinhole")


@pytest.fixture(scope="module")
def rendered_board_image(board):
    """Grayscale render of the board, large enough for reliable detection."""
    return board.generateImage((1400, 900), marginSize=40)


@pytest.fixture(scope="module")
def fisheye_result(fisheye_detections, board):
    return calibrate_intrinsics(fisheye_detections, board, IMAGE_SIZE, model="fisheye")


# --------------------------------------------------------------------------
# ARUCO_DICTS
# --------------------------------------------------------------------------


def test_aruco_dicts_covers_every_opencv_dict_name():
    expected = {name for name in dir(cv2.aruco) if name.startswith("DICT_")}
    assert expected, "sanity: cv2.aruco should expose DICT_* constants"
    missing = expected - set(ARUCO_DICTS)
    assert not missing, f"ARUCO_DICTS is missing {sorted(missing)}"


def test_aruco_dicts_values_match_opencv():
    for name, value in ARUCO_DICTS.items():
        assert value == getattr(cv2.aruco, name)
    assert ARUCO_DICTS["DICT_5X5_1000"] == cv2.aruco.DICT_5X5_1000


# --------------------------------------------------------------------------
# CharucoBoardSpec
# --------------------------------------------------------------------------


def test_spec_defaults_match_contract():
    spec = CharucoBoardSpec()
    assert spec.squares_x == 14
    assert spec.squares_y == 9
    assert spec.square_length == pytest.approx(0.04)
    assert spec.marker_length == pytest.approx(0.03)
    assert spec.aruco_dict == "DICT_5X5_1000"
    assert spec.legacy_pattern is False


def test_num_chessboard_corners_is_104(board_spec):
    assert board_spec.num_chessboard_corners == 104


def test_board_chessboard_corners_row_count_matches_spec(board_spec, board):
    """Pins the (X, Y) ordering: 14 columns x 9 rows -> 13*8 inner corners."""
    corners = board.getChessboardCorners()
    assert corners.shape[0] == board_spec.num_chessboard_corners == 104
    assert corners.shape[1] == 3


def test_build_returns_charuco_board(board):
    assert isinstance(board, cv2.aruco.CharucoBoard)
    assert tuple(board.getChessboardSize()) == (SQUARES_X, SQUARES_Y)
    assert board.getSquareLength() == pytest.approx(SQUARE_LENGTH, abs=1e-6)
    assert board.getMarkerLength() == pytest.approx(MARKER_LENGTH, abs=1e-6)


def test_build_rejects_marker_not_smaller_than_square():
    with pytest.raises(ValueError):
        CharucoBoardSpec(square_length=0.04, marker_length=0.04).build()
    with pytest.raises(ValueError):
        CharucoBoardSpec(square_length=0.04, marker_length=0.05).build()


def test_build_rejects_unknown_dict_name():
    with pytest.raises(ValueError):
        CharucoBoardSpec(aruco_dict="DICT_9X9_9000").build()


def test_build_rejects_too_few_squares_x():
    with pytest.raises(ValueError):
        CharucoBoardSpec(squares_x=1).build()


def test_build_rejects_too_few_squares_y():
    with pytest.raises(ValueError):
        CharucoBoardSpec(squares_y=1).build()


# --------------------------------------------------------------------------
# Detection dataclass
# --------------------------------------------------------------------------


def test_detection_num_corners_and_default_source():
    det = _make_detection([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0, 1, 2])
    assert det.num_corners == 3
    assert det.source == ""
    assert det.image_shape == IMAGE_SHAPE


# --------------------------------------------------------------------------
# detect_charuco
# --------------------------------------------------------------------------


def test_detect_charuco_on_rendered_board(rendered_board_image, board, board_spec):
    det = detect_charuco(rendered_board_image, board)
    assert det is not None
    # A clean, undistorted render should recover nearly every inner corner.
    assert det.num_corners >= int(0.9 * board_spec.num_chessboard_corners)
    assert det.num_corners <= board_spec.num_chessboard_corners


def test_detect_charuco_output_shapes_and_dtypes(rendered_board_image, board):
    det = detect_charuco(rendered_board_image, board)
    assert det is not None
    n = det.num_corners
    assert det.charuco_corners.shape == (n, 1, 2)
    assert det.charuco_corners.dtype == np.float32
    assert det.charuco_ids.shape == (n, 1)
    assert det.charuco_ids.dtype == np.int32
    assert det.image_shape == rendered_board_image.shape[:2]


def test_detect_charuco_ids_are_valid_and_unique(
    rendered_board_image, board, board_spec
):
    det = detect_charuco(rendered_board_image, board)
    assert det is not None
    ids = det.charuco_ids.reshape(-1)
    assert ids.min() >= 0
    assert ids.max() < board_spec.num_chessboard_corners
    assert len(np.unique(ids)) == len(ids)


def test_detect_charuco_returns_none_on_blank_image(board):
    blank = np.full((480, 640), 255, dtype=np.uint8)
    assert detect_charuco(blank, board) is None


def test_detect_charuco_returns_none_on_noise_image(board):
    rng = np.random.default_rng(20260728)
    noise = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
    assert detect_charuco(noise, board) is None


def test_detect_charuco_accepts_gray_and_bgr(rendered_board_image, board):
    gray = rendered_board_image
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert gray.ndim == 2
    assert bgr.shape == (gray.shape[0], gray.shape[1], 3)

    det_gray = detect_charuco(gray, board)
    det_bgr = detect_charuco(bgr, board)
    assert det_gray is not None
    assert det_bgr is not None
    assert det_gray.num_corners == det_bgr.num_corners
    assert det_gray.image_shape == det_bgr.image_shape == gray.shape[:2]
    np.testing.assert_allclose(
        det_gray.charuco_corners, det_bgr.charuco_corners, atol=1e-3
    )


def test_detect_charuco_respects_min_corners(rendered_board_image, board):
    assert detect_charuco(rendered_board_image, board, min_corners=8) is not None
    # more corners than the board even has -> must return None, not raise
    assert detect_charuco(rendered_board_image, board, min_corners=500) is None


# --------------------------------------------------------------------------
# calibrate_intrinsics -- synthetic round trips
# --------------------------------------------------------------------------


def test_fisheye_round_trip_recovers_ground_truth(fisheye_result, fisheye_detections):
    res = fisheye_result
    assert isinstance(res, CalibrationResult)
    assert res.model == "fisheye"
    assert res.num_views == len(fisheye_detections)
    assert res.image_size == IMAGE_SIZE

    assert res.K.shape == (3, 3)
    assert res.K.dtype == np.float64
    assert res.D.shape == (4,)
    assert res.D.dtype == np.float64

    for name, got, gt in [
        ("fx", res.K[0, 0], FISHEYE_K_GT[0, 0]),
        ("fy", res.K[1, 1], FISHEYE_K_GT[1, 1]),
        ("cx", res.K[0, 2], FISHEYE_K_GT[0, 2]),
        ("cy", res.K[1, 2], FISHEYE_K_GT[1, 2]),
    ]:
        rel = abs(got - gt) / abs(gt)
        assert rel < REL_TOL, f"{name}: got {got}, expected {gt} (rel={rel:.3e})"

    assert res.rms < RMS_TOL


def test_fisheye_round_trip_recovers_distortion(fisheye_result):
    np.testing.assert_allclose(fisheye_result.D, FISHEYE_D_GT, atol=5e-3)


def test_fisheye_per_view_errors(fisheye_result):
    errs = fisheye_result.per_view_errors
    assert errs.shape == (fisheye_result.num_views,)
    assert errs.dtype == np.float64
    assert np.all(np.isfinite(errs))
    assert np.all(errs >= 0.0)
    assert float(errs.max()) < RMS_TOL


def test_fisheye_result_carries_board_geometry(fisheye_result):
    spec = fisheye_result.board_spec
    assert spec.squares_x == SQUARES_X
    assert spec.squares_y == SQUARES_Y
    assert spec.square_length == pytest.approx(SQUARE_LENGTH, abs=1e-6)
    assert spec.marker_length == pytest.approx(MARKER_LENGTH, abs=1e-6)


def test_pinhole_round_trip_recovers_ground_truth(pinhole_detections, board):
    res = calibrate_intrinsics(pinhole_detections, board, IMAGE_SIZE, model="pinhole")
    assert res.model == "pinhole"
    assert res.num_views == len(pinhole_detections)
    assert res.image_size == IMAGE_SIZE

    assert res.K.shape == (3, 3)
    assert res.K.dtype == np.float64
    # CALIB_RATIONAL_MODEL -> exactly 8 coefficients (k1 k2 p1 p2 k3 k4 k5 k6)
    assert res.D.shape == (8,)
    assert res.D.dtype == np.float64

    for name, got, gt in [
        ("fx", res.K[0, 0], PINHOLE_K_GT[0, 0]),
        ("fy", res.K[1, 1], PINHOLE_K_GT[1, 1]),
        ("cx", res.K[0, 2], PINHOLE_K_GT[0, 2]),
        ("cy", res.K[1, 2], PINHOLE_K_GT[1, 2]),
    ]:
        rel = abs(got - gt) / abs(gt)
        assert rel < REL_TOL, f"{name}: got {got}, expected {gt} (rel={rel:.3e})"

    assert res.rms < RMS_TOL
    assert res.per_view_errors.shape == (res.num_views,)
    assert float(res.per_view_errors.max()) < RMS_TOL


# --------------------------------------------------------------------------
# calibrate_intrinsics -- validation / robustness
# --------------------------------------------------------------------------


def test_calibrate_accepts_explicit_board_spec(fisheye_detections, board, board_spec):
    """An explicit spec must be carried through verbatim, not re-derived.

    Re-deriving from the cv2 board loses precision (float32) and cannot
    recover the aruco dict *name*, so this is the discriminating check.
    """
    res = calibrate_intrinsics(
        fisheye_detections,
        board,
        IMAGE_SIZE,
        model="fisheye",
        board_spec=board_spec,
    )
    assert res.board_spec.squares_x == SQUARES_X
    assert res.board_spec.squares_y == SQUARES_Y
    assert res.board_spec.aruco_dict == ARUCO_DICT_NAME
    assert res.board_spec.square_length == SQUARE_LENGTH
    assert res.board_spec.marker_length == MARKER_LENGTH


def test_fix_k3_k4_defaults_to_leaving_them_free(fisheye_result):
    """Default fix_k3_k4=False must still fit k3 (ground truth k3 = -0.001)."""
    assert fisheye_result.D[2] != 0.0


def test_fix_k3_k4_pins_high_order_fisheye_coefficients(fisheye_detections, board):
    res = calibrate_intrinsics(
        fisheye_detections, board, IMAGE_SIZE, model="fisheye", fix_k3_k4=True
    )
    assert res.D.shape == (4,)
    assert res.D[2] == 0.0
    assert res.D[3] == 0.0
    # the reduced model must still fit this near-ideal data and locate the
    # principal point / focal length to the contract tolerance
    assert res.rms < RMS_TOL
    for name, got, gt in [
        ("fx", res.K[0, 0], FISHEYE_K_GT[0, 0]),
        ("fy", res.K[1, 1], FISHEYE_K_GT[1, 1]),
        ("cx", res.K[0, 2], FISHEYE_K_GT[0, 2]),
        ("cy", res.K[1, 2], FISHEYE_K_GT[1, 2]),
    ]:
        rel = abs(got - gt) / abs(gt)
        assert rel < REL_TOL, f"{name}: got {got}, expected {gt} (rel={rel:.3e})"


def test_fix_k3_k4_is_a_noop_for_pinhole(pinhole_detections, board):
    """Documented behaviour: fix_k3_k4 is ignored when model='pinhole'."""
    free = calibrate_intrinsics(
        pinhole_detections, board, IMAGE_SIZE, model="pinhole", fix_k3_k4=False
    )
    fixed = calibrate_intrinsics(
        pinhole_detections, board, IMAGE_SIZE, model="pinhole", fix_k3_k4=True
    )
    assert np.allclose(free.K, fixed.K)
    assert np.allclose(free.D, fixed.D)
    assert np.allclose(free.rms, fixed.rms)
    assert free.D.shape == fixed.D.shape == (8,)
    # the rational model's higher-order terms must not be pinned to zero
    assert fixed.D[4] != 0.0
    assert fixed.D[5] != 0.0


# --------------------------------------------------------------------------
# CalibrationResult.sources -- per-view provenance
# --------------------------------------------------------------------------


def test_sources_are_index_aligned_with_per_view_errors(
    fisheye_result, fisheye_detections
):
    res = fisheye_result
    assert len(res.sources) == res.num_views
    assert len(res.sources) == len(res.per_view_errors)
    assert list(res.sources) == [d.source for d in fisheye_detections]


def test_sources_reflect_only_surviving_views(fisheye_detections, board):
    """Dropped (<6 corner) views must not leave holes in the provenance list."""
    bad_a = _make_detection(
        [[10.0, 10.0], [20.0, 12.0], [30.0, 14.0]], [0, 1, 2], source="bad_a"
    )
    bad_b = _make_detection(
        [[11.0, 11.0], [21.0, 13.0], [31.0, 15.0], [41.0, 17.0], [51.0, 19.0]],
        [0, 1, 2, 3, 4],
        source="bad_b",
    )
    # interleave the doomed views so a naive "truncate the tail" would be caught
    mixed = list(fisheye_detections)
    mixed.insert(9, bad_b)
    mixed.insert(3, bad_a)

    res = calibrate_intrinsics(mixed, board, IMAGE_SIZE, model="fisheye")

    assert len(res.sources) == res.num_views == len(fisheye_detections)
    assert "bad_a" not in res.sources
    assert "bad_b" not in res.sources
    assert list(res.sources) == [d.source for d in fisheye_detections]


def test_calibrate_raises_below_default_min_views(fisheye_detections, board):
    with pytest.raises(ValueError):
        calibrate_intrinsics(fisheye_detections[:5], board, IMAGE_SIZE)


def test_calibrate_raises_below_explicit_min_views(fisheye_detections, board):
    with pytest.raises(ValueError):
        calibrate_intrinsics(fisheye_detections[:8], board, IMAGE_SIZE, min_views=10)


def test_calibrate_raises_on_empty_detections(board):
    with pytest.raises(ValueError):
        calibrate_intrinsics([], board, IMAGE_SIZE)


def test_calibrate_raises_on_unknown_model(fisheye_detections, board):
    with pytest.raises(ValueError):
        calibrate_intrinsics(fisheye_detections, board, IMAGE_SIZE, model="equidistant")


def test_calibrate_drops_views_with_too_few_corners(fisheye_detections, board):
    """Views with < 6 charuco corners are dropped, not crashed on."""
    degenerate = [
        _make_detection(
            [[10.0, 10.0], [20.0, 12.0], [30.0, 14.0]], [0, 1, 2], source="bad_a"
        ),
        _make_detection(
            [[11.0, 11.0], [21.0, 13.0], [31.0, 15.0], [41.0, 17.0], [51.0, 19.0]],
            [0, 1, 2, 3, 4],
            source="bad_b",
        ),
    ]
    mixed = list(fisheye_detections) + degenerate
    res = calibrate_intrinsics(mixed, board, IMAGE_SIZE, model="fisheye")

    assert res.num_views == len(fisheye_detections)
    assert res.per_view_errors.shape == (len(fisheye_detections),)
    assert res.rms < RMS_TOL
    rel = abs(res.K[0, 0] - FISHEYE_K_GT[0, 0]) / FISHEYE_K_GT[0, 0]
    assert rel < REL_TOL


def test_calibrate_raises_when_too_few_views_survive_corner_filter(board):
    """Enough detections to pass min_views, but all are below 6 corners."""
    degenerate = [
        _make_detection(
            [[10.0 + i, 10.0], [20.0 + i, 12.0], [30.0 + i, 14.0]],
            [0, 1, 2],
            source=f"bad_{i}",
        )
        for i in range(10)
    ]
    with pytest.raises(ValueError):
        calibrate_intrinsics(degenerate, board, IMAGE_SIZE, model="fisheye")


# --------------------------------------------------------------------------
# CalibrationResult.to_dict
# --------------------------------------------------------------------------


def test_to_dict_survives_json_round_trip(fisheye_result):
    d = fisheye_result.to_dict()
    assert isinstance(d, dict)
    text = json.dumps(d)  # must not raise
    back = json.loads(text)

    np.testing.assert_array_equal(
        np.asarray(back["K"], dtype=np.float64), fisheye_result.K
    )
    np.testing.assert_array_equal(
        np.asarray(back["D"], dtype=np.float64), fisheye_result.D
    )
    assert back["model"] == "fisheye"
    assert back["num_views"] == fisheye_result.num_views
    assert back["rms"] == pytest.approx(float(fisheye_result.rms), rel=0, abs=0)
    assert list(back["image_size"]) == list(IMAGE_SIZE)


def test_to_dict_contains_no_numpy_objects(fisheye_result):
    d = fisheye_result.to_dict()

    def walk(value, path="root"):
        assert not isinstance(value, np.ndarray), f"{path} is still an ndarray"
        assert not isinstance(value, np.generic), f"{path} is still a numpy scalar"
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")

    walk(d)
    assert isinstance(d["K"], list)
    assert len(d["K"]) == 3 and len(d["K"][0]) == 3
    assert isinstance(d["D"], list) and len(d["D"]) == 4
    assert isinstance(d["per_view_errors"], list)
    assert len(d["per_view_errors"]) == fisheye_result.num_views
    assert isinstance(d["rms"], float)


def test_to_dict_serializes_board_spec(fisheye_result):
    """The board geometry must survive serialization, keyed by its field name."""
    d = fisheye_result.to_dict()
    assert "board_spec" in d, f"to_dict() keys were {sorted(d)}"
    spec = d["board_spec"]
    assert isinstance(spec, dict)
    assert spec["squares_x"] == SQUARES_X
    assert spec["squares_y"] == SQUARES_Y
    assert spec["square_length"] == pytest.approx(SQUARE_LENGTH, abs=1e-6)
    assert spec["marker_length"] == pytest.approx(MARKER_LENGTH, abs=1e-6)
    assert spec["aruco_dict"] == ARUCO_DICT_NAME


# --------------------------------------------------------------------------
# sharpness_score
# --------------------------------------------------------------------------


def test_sharpness_score_prefers_sharp_board_render(rendered_board_image):
    sharp = rendered_board_image
    blurred = cv2.GaussianBlur(sharp, (9, 9), 3.0)
    s_sharp = sharpness_score(sharp)
    s_blur = sharpness_score(blurred)
    assert isinstance(s_sharp, float)
    assert isinstance(s_blur, float)
    assert s_sharp > s_blur


def test_sharpness_score_prefers_sharp_noise_image():
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (11, 11), 5.0)
    assert sharpness_score(sharp) > sharpness_score(blurred)


def test_sharpness_score_accepts_bgr(rendered_board_image):
    bgr = cv2.cvtColor(rendered_board_image, cv2.COLOR_GRAY2BGR)
    blurred_bgr = cv2.GaussianBlur(bgr, (9, 9), 3.0)
    assert sharpness_score(bgr) > sharpness_score(blurred_bgr)


# --------------------------------------------------------------------------
# coverage_grid / coverage_fraction
# --------------------------------------------------------------------------

GRID = (8, 6)  # (cols, rows)
CELL_W = IMAGE_W / GRID[0]
CELL_H = IMAGE_H / GRID[1]
NUM_CELLS = GRID[0] * GRID[1]


def _cell_center(row, col):
    return [(col + 0.5) * CELL_W, (row + 0.5) * CELL_H]


def _detection_covering(cells, source=""):
    pts = [_cell_center(r, c) for (r, c) in cells]
    return _make_detection(pts, list(range(len(pts))), source=source)


def test_coverage_grid_shape_and_dtype():
    det = _detection_covering([(0, 0)])
    grid = coverage_grid([det], IMAGE_SIZE, grid=GRID)
    assert grid.shape == (GRID[1], GRID[0])
    assert grid.dtype == np.bool_


def test_coverage_grid_shape_follows_grid_argument():
    det = _detection_covering([(0, 0)])
    grid = coverage_grid([det], IMAGE_SIZE, grid=(4, 3))
    assert grid.shape == (3, 4)


def test_coverage_fraction_zero_for_empty_detections():
    assert coverage_fraction([], IMAGE_SIZE, grid=GRID) == 0.0
    grid = coverage_grid([], IMAGE_SIZE, grid=GRID)
    assert grid.shape == (GRID[1], GRID[0])
    assert not grid.any()


def test_coverage_grid_marks_the_hit_cell():
    det = _detection_covering([(2, 5)])
    grid = coverage_grid([det], IMAGE_SIZE, grid=GRID)
    assert grid[2, 5]
    assert grid.sum() == 1


def test_coverage_fraction_increases_as_detections_spread():
    d0 = _detection_covering([(0, 0)], source="corner")
    d1 = _detection_covering([(0, c) for c in range(GRID[0])], source="top_row")
    d2 = _detection_covering(
        [(r, c) for r in range(GRID[1]) for c in range(0, GRID[0], 2)],
        source="half",
    )

    f0 = coverage_fraction([d0], IMAGE_SIZE, grid=GRID)
    f1 = coverage_fraction([d0, d1], IMAGE_SIZE, grid=GRID)
    f2 = coverage_fraction([d0, d1, d2], IMAGE_SIZE, grid=GRID)

    assert f0 == pytest.approx(1.0 / NUM_CELLS)
    assert f1 == pytest.approx(GRID[0] / NUM_CELLS)
    assert f0 < f1 < f2
    assert 0.0 <= f2 <= 1.0


def test_coverage_fraction_is_one_when_every_cell_hit():
    det = _detection_covering(
        [(r, c) for r in range(GRID[1]) for c in range(GRID[0])], source="full"
    )
    grid = coverage_grid([det], IMAGE_SIZE, grid=GRID)
    assert grid.all()
    assert coverage_fraction([det], IMAGE_SIZE, grid=GRID) == pytest.approx(1.0)


def test_coverage_fraction_matches_grid_mean():
    dets = [
        _detection_covering([(0, 0), (1, 3)], source="a"),
        _detection_covering([(5, 7)], source="b"),
    ]
    grid = coverage_grid(dets, IMAGE_SIZE, grid=GRID)
    assert coverage_fraction(dets, IMAGE_SIZE, grid=GRID) == pytest.approx(grid.mean())
