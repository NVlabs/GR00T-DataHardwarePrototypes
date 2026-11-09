# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ChArUco-based camera intrinsics + distortion calibration.

Pure OpenCV/numpy — no depthai, no hardware. The live-capture CLI lives in
``real_scripts/camera_calibration/calibrate_intrinsics.py``; everything here is
importable and unit-testable offline.

Two lens models are supported:

``fisheye``
    Kannala-Brandt equidistant (``cv2.fisheye``), 4 coefficients ``[k1 k2 k3 k4]``.
    Correct for the wide-FOV OAK-FFC modules.
``pinhole``
    Brown-Conrady with the rational extension (``CALIB_RATIONAL_MODEL``),
    8 coefficients ``[k1 k2 p1 p2 k3 k4 k5 k6]``.

Note that this repo has no pre-existing on-disk intrinsics schema — cameras pull
K live from device EEPROM at pipeline-build time — so ``CalibrationResult.to_dict``
defines a new format rather than conforming to one.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from threadpoolctl import threadpool_limits

    HAVE_THREADPOOLCTL = True
except ImportError:  # pragma: no cover - only in an environment missing a dependency
    from contextlib import nullcontext

    HAVE_THREADPOOLCTL = False

    def threadpool_limits(limits=None, user_api=None):
        return nullcontext()

    # Limit BLAS threads during calibration to prevent severe oversubscription.
    warnings.warn(
        "threadpoolctl is not installed; camera calibration solves will not pin "
        "the BLAS thread pool and may take minutes instead of seconds. "
        "Install it with `uv add threadpoolctl`.",
        RuntimeWarning,
        stacklevel=2,
    )

# All predefined ArUco dictionaries exposed by the installed OpenCV build,
# keyed by their canonical name (e.g. "DICT_5X5_1000").
ARUCO_DICTS: Dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}

# Drop views with too few matched corners to constrain a reliable pose.
MIN_CORNERS_PER_VIEW = 6

# Treat high reprojection error as a failed solve even when optimization converged.
MAX_PLAUSIBLE_RMS = 5.0

# Probe focal seeds at reduced precision, then solve the best seed to full convergence.
SEARCH_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1e-4)


@dataclass
class CharucoBoardSpec:
    """Printed board geometry. Lengths in meters.

    ``squares_x`` counts squares along X (columns) and ``squares_y`` along Y
    (rows), matching the ``(squares_x, squares_y)`` ordering used by
    ``cv2.aruco.CharucoBoard`` and by ``generate_charuco_board.py``.
    """

    squares_x: int = 14
    squares_y: int = 9
    square_length: float = 0.04
    marker_length: float = 0.03
    aruco_dict: str = "DICT_5X5_1000"
    legacy_pattern: bool = False

    @property
    def num_chessboard_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def build(self) -> Any:
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError(
                f"Board must be at least 2x2 squares, got "
                f"{self.squares_x}x{self.squares_y}."
            )
        if self.square_length <= 0 or self.marker_length <= 0:
            raise ValueError(
                f"square_length and marker_length must be positive, got "
                f"{self.square_length} and {self.marker_length}."
            )
        if self.marker_length >= self.square_length:
            raise ValueError(
                f"marker_length ({self.marker_length}) must be smaller than "
                f"square_length ({self.square_length})."
            )
        if self.aruco_dict not in ARUCO_DICTS:
            raise ValueError(
                f"Unknown ArUco dictionary {self.aruco_dict!r}. "
                f"Expected one of {sorted(ARUCO_DICTS)}."
            )
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[self.aruco_dict])
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            float(self.square_length),
            float(self.marker_length),
            dictionary,
        )
        # Boards printed by OpenCV < 4.6 use the opposite marker/square parity.
        if self.legacy_pattern:
            board.setLegacyPattern(True)
        return board

    def to_dict(self) -> Dict[str, Any]:
        return {
            "squares_x": int(self.squares_x),
            "squares_y": int(self.squares_y),
            "square_length": float(self.square_length),
            "marker_length": float(self.marker_length),
            "aruco_dict": self.aruco_dict,
            "legacy_pattern": bool(self.legacy_pattern),
        }


@dataclass
class Detection:
    """ChArUco corners found in one view."""

    charuco_corners: np.ndarray  # (N, 1, 2) float32
    charuco_ids: np.ndarray  # (N, 1) int32
    image_shape: Tuple[int, int]  # (height, width)
    source: str = ""

    @property
    def num_corners(self) -> int:
        return int(self.charuco_ids.shape[0])


def detect_charuco(
    image: np.ndarray,
    board: Any,
    min_corners: int = 8,
    source: str = "",
) -> Optional[Detection]:
    """Detect ChArUco corners in a BGR or grayscale image.

    Returns ``None`` when fewer than ``min_corners`` corners are recovered.
    """
    if image is None:
        return None
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            f"Expected (H,W) grayscale or (H,W,3) BGR image, got shape {image.shape}."
        )

    detector_params = cv2.aruco.DetectorParameters()
    # Subpixel refinement is what makes the corners accurate enough to calibrate.
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco_params = cv2.aruco.CharucoParameters()
    # Interpolate a corner even when only one of its two neighbour markers is
    # visible — keeps board edges usable, which is where distortion shows up.
    charuco_params.minMarkers = 1

    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None

    return Detection(
        charuco_corners=np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2),
        charuco_ids=np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1),
        image_shape=(int(gray.shape[0]), int(gray.shape[1])),
        source=source,
    )


@dataclass
class CalibrationResult:
    model: str
    K: np.ndarray  # (3, 3) float64
    D: np.ndarray  # (4,) fisheye or (8,) pinhole, float64
    image_size: Tuple[int, int]  # (width, height)
    rms: float
    per_view_errors: np.ndarray  # (num_views,) float64, mean px error per view
    num_views: int
    board_spec: CharucoBoardSpec
    sources: List[str] = field(default_factory=list)
    # Focal-length seed that converged (fisheye only); None for pinhole.
    init_focal_seed: Optional[float] = None
    # Views dropped because they abort fisheye initialisation on their own.
    dropped_views: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        K = np.asarray(self.K, dtype=np.float64)
        return {
            "model": self.model,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "K": K.tolist(),
            "D": np.asarray(self.D, dtype=np.float64).ravel().tolist(),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "rms": float(self.rms),
            "per_view_errors": np.asarray(self.per_view_errors, dtype=np.float64)
            .ravel()
            .tolist(),
            "num_views": int(self.num_views),
            "board_spec": self.board_spec.to_dict(),
            "sources": list(self.sources),
            "init_focal_seed": (
                None if self.init_focal_seed is None else float(self.init_focal_seed)
            ),
            "dropped_views": list(self.dropped_views),
        }


def _match_points(
    detections: List[Detection], board: Any
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Detection]]:
    """Map ChArUco ids to board object points, dropping under-constrained views."""
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    used: List[Detection] = []
    for det in detections:
        if det.num_corners < MIN_CORNERS_PER_VIEW:
            continue
        matched = board.matchImagePoints(det.charuco_corners, det.charuco_ids)
        if matched is None:
            continue
        op, ip = matched
        if op is None or ip is None or len(op) < MIN_CORNERS_PER_VIEW:
            continue
        obj_points.append(np.asarray(op, dtype=np.float64).reshape(-1, 1, 3))
        img_points.append(np.asarray(ip, dtype=np.float64).reshape(-1, 1, 2))
        used.append(det)
    return obj_points, img_points, used


def _fisheye_seed_candidates(width: int, height: int) -> List[float]:
    """Focal-length seeds to try, widest lens first.

    ``cv2.fisheye.calibrate`` initialises extrinsics per view by undistorting
    corners with the seed intrinsics. If the seed focal length is too short, a
    corner sits beyond theta = pi/2 where the tangent flips sign, the per-view
    homography collapses, and the solve dies on an assertion inside
    InitExtrinsics. Which seeds survive is not monotonic, so try a ladder and
    take the first that converges — on real data every seed that converges
    reaches the same optimum.
    """
    m = float(max(width, height))
    return [m / r for r in (3.2, 2.5, 4.0, 2.0, 5.0, 1.6, 6.5, 1.3, 8.0, 1.0)]


def _seeded_K(seed: float, width: int, height: int) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = float(seed)
    K[0, 2] = (width - 1) / 2.0
    K[1, 2] = (height - 1) / 2.0
    return K


def _fisheye_attempt(obj_points, img_points, image_size, flags, criteria, seeds):
    """Solve from every seed and keep the best.

    Seeds that merely fail to raise are not good enough — some settle into a
    local minimum with a reprojection error of tens of pixels. Return the lowest
    RMS across all seeds, and only if it is physically plausible.
    """
    width, height = image_size
    best = None
    best_seed = None
    # Reject divergent or poorly fitting seeds before the full-precision solve.
    for seed in seeds:
        try:
            solved = cv2.fisheye.calibrate(
                obj_points,
                img_points,
                (width, height),
                _seeded_K(seed, width, height),
                np.zeros((4, 1), dtype=np.float64),
                flags=flags,
                criteria=SEARCH_CRITERIA,
            )
        except cv2.error:
            continue
        rms = float(solved[0])
        if not np.isfinite(rms) or rms > MAX_PLAUSIBLE_RMS:
            continue
        if best is None or rms < float(best[0]):
            best, best_seed = solved, float(seed)
    if best is None:
        return None, None
    # Re-solve the winning seed to full precision.
    try:
        refined = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            (width, height),
            _seeded_K(best_seed, width, height),
            np.zeros((4, 1), dtype=np.float64),
            flags=flags,
            criteria=criteria,
        )
        rms = float(refined[0])
        if np.isfinite(rms) and rms <= MAX_PLAUSIBLE_RMS:
            return refined, best_seed
    except cv2.error:
        pass
    return best, best_seed


def _max_corner_radius(img_pts: np.ndarray, width: int, height: int) -> float:
    pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    return float(
        np.max(np.hypot(pts[:, 0] - (width - 1) / 2.0, pts[:, 1] - (height - 1) / 2.0))
    )


def _fisheye_per_view_errors(obj_points, img_points, K, D, rvecs, tvecs) -> np.ndarray:
    errors = []
    for op, ip, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(
            op.reshape(-1, 1, 3), rvec, tvec, K, D.reshape(4, 1)
        )
        diff = proj.reshape(-1, 2) - ip.reshape(-1, 2)
        errors.append(float(np.sqrt((diff**2).sum(axis=1)).mean()))
    return np.asarray(errors, dtype=np.float64)


def _spec_from_board(board: Any) -> CharucoBoardSpec:
    """Best-effort spec recovery for callers that only pass a cv2 board.

    The board stores lengths as float32, so values round-trip approximately;
    pass ``board_spec`` explicitly when exact values matter.
    """
    squares_x, squares_y = board.getChessboardSize()
    return CharucoBoardSpec(
        squares_x=int(squares_x),
        squares_y=int(squares_y),
        square_length=float(board.getSquareLength()),
        marker_length=float(board.getMarkerLength()),
        legacy_pattern=bool(board.getLegacyPattern()),
    )


def calibrate_intrinsics(
    detections: List[Detection],
    board: Any,
    image_size: Tuple[int, int],
    model: str = "fisheye",
    min_views: int = 6,
    board_spec: Optional[CharucoBoardSpec] = None,
    fix_k3_k4: bool = False,
) -> CalibrationResult:
    """Solve for K and distortion from a set of ChArUco detections.

    ``image_size`` is ``(width, height)``.

    ``fix_k3_k4`` pins the two highest-order fisheye terms to zero. On mildly
    distorted lenses, or when corners do not reach the frame edges, those terms
    overfit and can yield a degenerate undistort matrix; pinning them typically
    leaves RMS and K unchanged. Ignored for ``model="pinhole"``.
    """
    if model not in ("fisheye", "pinhole"):
        raise ValueError(f"model must be 'fisheye' or 'pinhole', got {model!r}.")
    if len(detections) < min_views:
        raise ValueError(
            f"Need at least {min_views} views to calibrate, got {len(detections)}."
        )

    obj_points, img_points, used = _match_points(detections, board)
    if len(obj_points) < min_views:
        raise ValueError(
            f"Only {len(obj_points)} of {len(detections)} views had at least "
            f"{MIN_CORNERS_PER_VIEW} matched corners; need {min_views}."
        )

    width, height = int(image_size[0]), int(image_size[1])
    init_focal_seed = None
    dropped_views: List[str] = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)

    # Limit BLAS to one thread for the per-view normal-equation solve; large
    # view sets otherwise cause heavy oversubscription without improving results.
    with threadpool_limits(limits=1):
        if model == "fisheye":
            flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_FIX_SKEW
                | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            )
            if fix_k3_k4:
                flags |= cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
            seeds = _fisheye_seed_candidates(width, height)
            size = (width, height)
            solved, init_focal_seed = _fisheye_attempt(
                obj_points, img_points, size, flags, criteria, seeds
            )
            if solved is None:
                # Recomputing extrinsics can diverge for extreme-radius views as K
                # changes.
                # Remove the widest views incrementally until the calibration converges.
                radii = [_max_corner_radius(ip, width, height) for ip in img_points]
                order = list(np.argsort(radii))
                for n_drop in range(1, max(1, len(order) - min_views) + 1):
                    keep = sorted(order[: len(order) - n_drop])
                    if len(keep) < min_views:
                        break
                    solved, init_focal_seed = _fisheye_attempt(
                        [obj_points[i] for i in keep],
                        [img_points[i] for i in keep],
                        size,
                        flags,
                        criteria,
                        seeds,
                    )
                    if solved is not None:
                        dropped_views = [
                            used[i].source for i in order[len(order) - n_drop :]
                        ]
                        obj_points = [obj_points[i] for i in keep]
                        img_points = [img_points[i] for i in keep]
                        used = [used[i] for i in keep]
                        break
            if solved is None:
                raise RuntimeError(
                    f"Fisheye calibration failed to initialise from any of "
                    f"{len(seeds)} focal-length seeds "
                    f"({max(seeds):.0f}..{min(seeds):.0f} px), including after "
                    f"dropping views that fail on their own. The capture may be "
                    f"too extreme for the fisheye model; try --model pinhole to "
                    f"see whether it is usable at all."
                )
            rms, K, D, rvecs, tvecs = solved
            D = np.asarray(D, dtype=np.float64).ravel()[:4]
            per_view = _fisheye_per_view_errors(
                obj_points, img_points, K, D, rvecs, tvecs
            )
        else:
            # calibrateCameraExtended rejects float64 point lists, unlike
            # cv2.fisheye.calibrate which accepts either.
            (
                rms,
                K,
                D,
                _rvecs,
                _tvecs,
                _std_int,
                _std_ext,
                per_view,
            ) = cv2.calibrateCameraExtended(
                [np.ascontiguousarray(o, dtype=np.float32) for o in obj_points],
                [np.ascontiguousarray(i, dtype=np.float32) for i in img_points],
                (width, height),
                None,
                None,
                flags=cv2.CALIB_RATIONAL_MODEL,
                criteria=criteria,
            )
            D = np.asarray(D, dtype=np.float64).ravel()
            # The rational model returns 14 slots but only fills the first 8;
            # truncating is lossless only while the rest stay zero.
            if D[8:].any():
                raise RuntimeError(
                    f"Unexpected non-zero distortion terms beyond index 8: {D[8:]}. "
                    f"Truncating to 8 coefficients would silently lose them."
                )
            D = D[:8]
            per_view = np.asarray(per_view, dtype=np.float64).ravel()

    return CalibrationResult(
        model=model,
        K=np.asarray(K, dtype=np.float64).reshape(3, 3),
        D=D,
        image_size=(width, height),
        rms=float(rms),
        per_view_errors=per_view,
        num_views=len(obj_points),
        board_spec=board_spec if board_spec is not None else _spec_from_board(board),
        init_focal_seed=init_focal_seed if model == "fisheye" else None,
        dropped_views=dropped_views,
        sources=[d.source for d in used],
    )


def sharpness_score(image: np.ndarray) -> float:
    """Variance of the Laplacian — higher is sharper. Used to reject motion blur."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def coverage_grid(
    detections: List[Detection],
    image_size: Tuple[int, int],
    grid: Tuple[int, int] = (8, 6),
) -> np.ndarray:
    """Boolean ``(rows, cols)`` mask of which image cells have seen a corner.

    ``grid`` is ``(cols, rows)``; ``image_size`` is ``(width, height)``.
    """
    cols, rows = int(grid[0]), int(grid[1])
    width, height = float(image_size[0]), float(image_size[1])
    mask = np.zeros((rows, cols), dtype=bool)
    for det in detections:
        pts = np.asarray(det.charuco_corners, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            continue
        cx = np.clip((pts[:, 0] / width * cols).astype(int), 0, cols - 1)
        cy = np.clip((pts[:, 1] / height * rows).astype(int), 0, rows - 1)
        mask[cy, cx] = True
    return mask


def estimate_usable_radius(
    max_intensity: np.ndarray, dark_level: float = 25.0
) -> float:
    """Radius, in pixels, at which the lens image circle goes dark.

    Wide fisheye modules often project a circle smaller than the sensor, so the
    frame corners are permanently black and no target can ever be imaged there.
    Pass a per-pixel maximum over many views (anything inside the circle will
    have been bright in at least one of them); the return value is the largest
    radius still lit, or the frame-corner radius when the circle overfills.
    """
    height, width = max_intensity.shape[:2]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    corner = float(np.hypot(cx, cy))
    yy, xx = np.mgrid[0:height, 0:width]
    rad = np.hypot(xx - cx, yy - cy)
    step = 10.0
    r = step
    last_lit = corner
    while r < corner:
        band = (rad >= r - step) & (rad < r + step)
        if band.any() and float(np.percentile(max_intensity[band], 90)) < dark_level:
            return float(r - step)
        if band.any():
            last_lit = float(r)
        r += step
    return min(last_lit, corner)


def radial_fill(detections: List[Detection], image_size: Tuple[int, int]) -> float:
    """How far out, radially, the corners actually reach — 0 to 1.

    Distortion is a radial function, so what constrains it is the largest
    corner radius, not how many grid cells are touched. 1.0 means corners reach
    the image corner; 0.7 means the outer 30% of the radial range is fit by
    extrapolation only. This is the number to watch on a fisheye.
    """
    width, height = float(image_size[0]), float(image_size[1])
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    corner_r = float(np.hypot(cx, cy))
    if corner_r <= 0:
        return 0.0
    best = 0.0
    for det in detections:
        pts = np.asarray(det.charuco_corners, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            continue
        best = max(best, float(np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))))
    return min(best / corner_r, 1.0)


def coverage_fraction(
    detections: List[Detection],
    image_size: Tuple[int, int],
    grid: Tuple[int, int] = (8, 6),
) -> float:
    return float(coverage_grid(detections, image_size, grid).mean())
