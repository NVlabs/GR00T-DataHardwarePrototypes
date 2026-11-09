# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np

from umi.common.utility.pose import invert_transformation

# Suppress per-frame OpenCV warnings (e.g. drawFrameAxes axes out of frame).
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# Defaults shared by the config-driven dispatcher (detect_marker_pose)
DEFAULT_ARUCO_DICT = "DICT_4X4_50"
DEFAULT_CHARUCO_DICT = "DICT_4X4_250"


def _resolve_aruco_dict(value):
    """Accept either an OpenCV ArUco dict constant or its string name.

    Strings are resolved via ``getattr(cv2.aruco, value)`` so configs can
    specify e.g. ``"DICT_4X4_250"`` directly.
    """
    if isinstance(value, str):
        return getattr(cv2.aruco, value)
    return value


def _resolve_distortion(model, coeffs):
    """Validate a (distortion_model, dist_coeffs) pair; return it normalized.

    Returns ("none", zeros(5)) when no model is requested, so callers can pass
    the result straight to solvePnP.
    """
    model = (model or "none").lower()
    if model == "none":
        return "none", np.zeros(5)
    if model not in ("pinhole", "fisheye"):
        raise ValueError(
            f"Unknown distortion_model {model!r}; expected 'none', 'pinhole' or 'fisheye'"
        )
    # A valid projection model may have no fitted distortion coefficients.
    # Select the correct lens projection before considering coefficient refinement.
    if coeffs is None:
        return model, np.zeros(4 if model == "fisheye" else 5)
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    # Kannala-Brandt is exactly 4 terms. Silently accepting a rational-model
    # vector here would fit a completely different lens and still return a
    # plausible pose.
    if model == "fisheye" and coeffs.size != 4:
        raise ValueError(
            f"fisheye (Kannala-Brandt) expects 4 dist_coeffs, got {coeffs.size}. "
            "Rational/pinhole coefficients are NOT interchangeable with these."
        )
    if model == "pinhole" and coeffs.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"pinhole expects 4, 5, 8, 12 or 14 dist_coeffs, got {coeffs.size}"
        )
    return model, coeffs


def _draw_frame_axes(img, camera_matrix, model, dist_coeffs, rvec, tvec, length=0.05):
    """drawFrameAxes equivalent that honours a fisheye lens model.

    cv2.drawFrameAxes projects through the pinhole model only, so on a fisheye
    lens it draws axes that do not lie on the board.
    """
    if model != "fisheye":
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, length)
        return
    axes = np.array(
        [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=np.float64
    ).reshape(-1, 1, 3)
    pts, _ = cv2.fisheye.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)
    pts = pts.reshape(-1, 2)
    if not np.isfinite(pts).all():
        return
    origin = tuple(np.round(pts[0]).astype(int))
    for end, color in zip(pts[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(img, origin, tuple(np.round(end).astype(int)), color, 2)


def get_camera_position_and_orientation_charuco(
    frame_data,
    aruco_dict=DEFAULT_CHARUCO_DICT,
    squares_x: int = 6,
    squares_y: int = 8,
    square_length: float = 0.08325,  # meters
    marker_length: float = 0.00624375,  # meters
    min_markers: int = 6,
    viz: bool = True,
    verbose: bool = False,
    distortion_model: str = "none",
    dist_coeffs=None,
):
    """
    Get camera position and orientation using Charuco board detection.

    A Charuco board combines a chessboard with ArUco markers for more robust
    and accurate pose estimation compared to ArUco markers alone.

    Args:
        frame_data: Frame data containing rgb and intrinsics
        aruco_dict: ArUco dictionary (cv2 constant or string name, e.g.
            "DICT_4X4_250") used for markers embedded in the ChArUco board.
        squares_x: Number of squares in X direction on the board
        squares_y: Number of squares in Y direction on the board
        square_length: Length of each chessboard square in meters
        marker_length: Length of ArUco marker in meters (must be < square_length)
        min_markers: Minimum number of markers required for valid detection
        viz: Whether to visualize detection results
        distortion_model: Lens model used by the pose solve. ``"none"``
            (default) assumes no distortion. ``"pinhole"`` passes
            ``dist_coeffs`` to solvePnP as OpenCV rational coefficients
            (k1 k2 p1 p2 k3 ...). ``"fisheye"`` treats them as Kannala-Brandt
            (4 coeffs) and undistorts the corners before solving, since OpenCV
            has no fisheye-aware solvePnP.
        dist_coeffs: Distortion coefficients matching ``distortion_model``.
            Required unless the model is ``"none"``.

    Returns:
        T_AC: 4x4 transformation matrix (camera pose relative to charuco board center)
              Returns None if detection fails
        rgb_image: Visualization image with drawn axes (if viz=True) or original image
    """
    try:
        camera_matrix = frame_data.intrinsics
        # Use distortion carried with the frame unless explicit coefficients
        # override it for a separate calibration.
        if dist_coeffs is None:
            dist_coeffs = getattr(frame_data, "distortion", None)
        distortion_model, dist_coeffs = _resolve_distortion(
            distortion_model, dist_coeffs
        )
        # Corner interpolation below never uses a lens model (see CharucoParameters
        # note), so these are for the pose solve only.

        rgb_image = frame_data.rgb

        if rgb_image is None:
            print("Error: RGB image is None")
            return None, None

        # Require intrinsics before solvePnP so a calibration error cannot be
        # misreported as marker detection failure. Uncalibrated OAK-FFC modules
        # need a matching fallback or per-module calibration.
        if camera_matrix is None:
            print(
                "Error: camera has no intrinsics (K). The ChArUco board may be "
                "perfectly visible -- this is NOT a detection failure. Either "
                "the device has no EEPROM calibration, or its sensor/resolution "
                "has no entry in oak_device.FALLBACK_INTRINSICS. Calibrate it "
                "with real_scripts/camera_calibration/calibrate_intrinsics.py."
            )
            return None, None

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        aruco_dict_name = aruco_dict if isinstance(aruco_dict, str) else "<cv2 const>"
        aruco_dict = cv2.aruco.getPredefinedDictionary(_resolve_aruco_dict(aruco_dict))
        charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            aruco_dict,
        )

        charuco_params = cv2.aruco.CharucoParameters()
        # Leave ChArUco cameraMatrix and distCoeffs unset for local-homography
        # corner interpolation; a pinhole model can reject valid fisheye corners.
        # Use camera intrinsics later in the pose solve. Permit one visible neighbor
        # marker for partially occluded ChArUco corners.
        try:
            charuco_params.minMarkers = 1
            charuco_params.tryRefineMarkers = True
        except AttributeError:
            # Some supported OpenCV builds omit these optional fields.
            pass
        detector_params = cv2.aruco.DetectorParameters()

        charuco_detector = cv2.aruco.CharucoDetector(
            charuco_board, charuco_params, detector_params
        )

        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            charuco_detector.detectBoard(gray)
        )

        n_markers = 0 if marker_ids is None else len(marker_ids)
        n_ccorners = 0 if charuco_ids is None else len(charuco_ids)
        if verbose:
            print(
                f"[charuco] board={squares_x}x{squares_y} "
                f"sq={square_length} mk={marker_length} "
                f"-> markers={n_markers} charuco_corners={n_ccorners}"
            )

        # Markers detected but no ChArUco corners interpolated -> the printed
        # board's dict / legacy_pattern / marker_id_offset likely don't match.
        if n_markers > 0 and n_ccorners == 0:
            detected_ids = np.asarray(marker_ids).reshape(-1)
            min_id = int(detected_ids.min())
            max_id = int(detected_ids.max())
            print(
                f"[charuco] HINT: marker IDs {min_id}..{max_id} detected but "
                f"0 corners. Check dict ({aruco_dict_name!r}), legacy_pattern, "
                f"and marker_id_offset; regenerate via generate_charuco_board.py."
            )

        if viz:
            rgb_image = rgb_image.copy()
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(rgb_image, marker_corners, marker_ids)
            if charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    rgb_image, charuco_corners, charuco_ids
                )

        if marker_ids is None or len(marker_ids) < min_markers:
            print(f"Not enough ArUco markers detected: {n_markers}/{min_markers}")
            return None, rgb_image if viz else None

        # Need at least 4 Charuco corners for pose estimation (solvePnP requirement)
        min_corners_required = max(4, min_markers)  # solvePnP needs at least 4 points
        if charuco_ids is None or len(charuco_ids) < min_corners_required:
            print(
                f"Not enough Charuco corners detected: {0 if charuco_ids is None else len(charuco_ids)}/{min_corners_required}"
            )
            return None, rgb_image if viz else None

        obj_points, img_points = charuco_board.matchImagePoints(
            charuco_corners, charuco_ids
        )

        if obj_points is None or img_points is None:
            print("Error: matchImagePoints returned None")
            return None, rgb_image if viz else None

        n_points = len(obj_points) if obj_points.ndim > 1 else 0
        if n_points < 4:
            print(f"Not enough matched points for solvePnP: {n_points}/4")
            return None, rgb_image if viz else None

        # OpenCV solvePnP interprets distortion as a pinhole model. For fisheye
        # lenses, first map corners to normalized rays and solve with identity K;
        # passing fisheye coefficients directly would silently use the wrong model.
        if distortion_model == "fisheye":
            rays = cv2.fisheye.undistortPoints(
                img_points.reshape(-1, 1, 2), camera_matrix, dist_coeffs
            )
            success, rvec, tvec = cv2.solvePnP(obj_points, rays, np.eye(3), np.zeros(5))
        else:
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, camera_matrix, dist_coeffs
            )

        if not success:
            print("Error: solvePnP failed for Charuco board")
            return None, rgb_image if viz else None

        # Convert to 4x4 transformation matrix (T_CA: charuco in camera frame)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        T_CA = np.eye(4)
        T_CA[:3, :3] = rotation_matrix
        T_CA[:3, 3] = tvec.flatten()

        # Depth refinement (a 3D-3D Procrustes fit of back-projected corners) is
        # not applied: the pose comes from the PnP solve above alone.

        # Invert to get camera pose relative to charuco board (T_AC)
        T_AC = invert_transformation(T_CA)

        if viz:
            _draw_frame_axes(
                rgb_image, camera_matrix, distortion_model, dist_coeffs, rvec, tvec
            )

        return T_AC, rgb_image

    except Exception as e:
        print(f"Error in get_camera_position_and_orientation_charuco: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def get_camera_position_and_orientation(
    frame_data,
    marker_size,
    aruco_dict=DEFAULT_ARUCO_DICT,
    viz=False,
):
    """
    Get camera position and orientation from a single ArUco marker.

    The pose comes from `solvePnP` over the marker's own corner geometry and
    their 2D detections -- RGB and intrinsics, no depth. Depth, when the camera
    has it, is only a validity filter: a marker whose corners the depth sensor
    could not see is skipped rather than solved. So a camera without depth (a
    head OAK with `enable_depth` off) solves the same pose, unfiltered.

    Args:
        frame_data: Frame data carrying rgb and intrinsics, and optionally
            depth.
        marker_size: Physical size of the ArUco marker edge in meters.
        aruco_dict: ArUco dictionary (cv2 constant or string name, e.g.
            "DICT_4X4_50").
        viz: Whether to visualize detection results.
    """
    try:
        camera_matrix = frame_data.intrinsics
        dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

        rgb_image = frame_data.rgb
        # Depthless RGB frames have no depth attribute; treat it as unavailable.
        depth_image = getattr(frame_data, "depth", None)
        if depth_image is not None and depth_image.dtype == np.uint16:
            depth_image = (
                depth_image.astype(np.float32) / 1000.0
            )  # uint16 mm -> float32 m
        # Resize depth to RGB resolution if collected at native LiDAR resolution
        if (
            depth_image is not None
            and rgb_image is not None
            and depth_image.shape[:2] != rgb_image.shape[:2]
        ):
            depth_image = cv2.resize(
                depth_image,
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        if rgb_image is None:
            print("Error: RGB image is None")
            return None, None

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(_resolve_aruco_dict(aruco_dict))
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        if viz:
            for i, corner in enumerate(corners):
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                for j, point in enumerate(corner[0]):
                    point_int = tuple(point.astype(int))
                    cv2.circle(rgb_image, point_int, 5, colors[j], -1)

                    cv2.putText(
                        rgb_image,
                        str(j),
                        (point_int[0] + 10, point_int[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colors[j],
                        2,
                    )

        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                marker_corners = corners[i][0]

                # Require valid depth at all four corners because solvePnP receives four
                # marker points. Depth filters validity only; it is not an input to the
                # solve.
                # The positive-value test also rejects NaN.
                if depth_image is not None:
                    ij = marker_corners.astype(int)
                    if not np.all(depth_image[ij[:, 1], ij[:, 0]] > 0):
                        print("Error: marker corners without valid depth")
                        continue

                # Define the reference 3D positions of the marker's corners in the marker's own coordinate system
                marker_size_half = marker_size / 2.0
                reference_3d_corners = np.array(
                    [
                        [-marker_size_half, marker_size_half, 0],
                        [marker_size_half, marker_size_half, 0],
                        [marker_size_half, -marker_size_half, 0],
                        [-marker_size_half, -marker_size_half, 0],
                    ],
                    dtype=np.float32,
                )

                success, rvec, tvec = cv2.solvePnP(
                    reference_3d_corners,
                    marker_corners.astype(np.float32),
                    camera_matrix,
                    dist_coeffs,
                )

                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rvec)

                    # Invert the rotation and translation to get the camera's pose relative to the marker
                    T_CA = np.eye(4)
                    T_CA[:3, :3] = rotation_matrix
                    T_CA[:3, -1] = tvec[:, 0]
                    T_CA[3, 3] = 1
                    T_AC = invert_transformation(T_CA)
                    cv2.drawFrameAxes(
                        rgb_image, camera_matrix, dist_coeffs, rvec, tvec, 0.05
                    )
                    return T_AC, rgb_image
                else:
                    print("Error: solvePnP failed")
                    return None, None

        else:
            print("No ArUco markers detected")
            return None, None

        return None, None

    except Exception as e:
        print(f"Error in get_camera_position_and_orientation: {e}")
        import traceback

        traceback.print_exc()
        return None, None


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

# Keys recognized in `marker_cfg` for each mode. ChArUco is a strict superset
# of ArUco: both share ``aruco_dict``; ChArUco adds board geometry fields.
_ARUCO_KEYS = {"aruco_dict", "marker_size"}
_CHARUCO_KEYS = {
    "aruco_dict",
    "squares_x",
    "squares_y",
    "square_length",
    "marker_length",
    "min_markers",
}
# Read distortion model and coefficients from the calibrated camera, not
# shared config: lens properties can differ between modules.


def detect_marker_pose(frame_data, marker_cfg: dict, viz: bool = False):
    """Unified ArUco/ChArUco pose estimation driven by a config dict.

    The config dict uses the following fields:

    - ``type``: ``"aruco"`` or ``"charuco"`` (required).
    - ``aruco_dict``: dictionary name or cv2 constant (optional; per-mode default).
    - ArUco-only: ``marker_size`` (meters; required when ``type == "aruco"``).
    - ChArUco-only: ``squares_x``, ``squares_y``, ``square_length``,
      ``marker_length``, ``min_markers`` (all optional; sensible defaults).

    Unknown keys are ignored so that callers can safely pass a superset dict
    (e.g. a raw ``aruco:`` config section).
    """
    mode = marker_cfg.get("type", "aruco")

    if mode == "charuco":
        kwargs = {k: marker_cfg[k] for k in _CHARUCO_KEYS if k in marker_cfg}
        return get_camera_position_and_orientation_charuco(
            frame_data, viz=viz, **kwargs
        )
    if mode == "aruco":
        if "marker_size" not in marker_cfg:
            raise ValueError(
                "marker_cfg with type='aruco' requires a 'marker_size' entry"
            )
        kwargs = {k: marker_cfg[k] for k in _ARUCO_KEYS if k in marker_cfg}
        return get_camera_position_and_orientation(frame_data, viz=viz, **kwargs)

    raise ValueError(f"Unknown marker_cfg type {mode!r}; expected 'aruco' or 'charuco'")
