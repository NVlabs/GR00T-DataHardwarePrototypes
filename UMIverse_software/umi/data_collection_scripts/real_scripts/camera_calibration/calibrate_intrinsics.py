#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ChArUco intrinsics + distortion calibration for OAK (AR0234) cameras.

Two subcommands, so calibration can be re-tuned without re-shooting the board:

Capture frames from a live OAK stream (auto-accepts sharp, novel views)::

    uv run real_scripts/camera_calibration/calibrate_intrinsics.py capture \
        --oak-device-name <oak-device-name> --socket CAM_B --resolution 640 400 \
        --output-dir /tmp/umi_intrinsics/right_umi_camera

Solve from any directory of images::

    uv run real_scripts/camera_calibration/calibrate_intrinsics.py calibrate \
        --image-dir /tmp/umi_intrinsics/right_umi_camera \
        --model fisheye --undistort-preview

Board defaults match the charuco board in
config/data_recording/production/bimanual_v4.yaml:
14x9 squares, 40 mm squares, 30 mm markers, DICT_5X5_1000.

real_scripts/data_collection/generate_charuco_board.py prints this board by default.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from umi.common.hardware_sdk.camera.intrinsics_calibration import (
    CharucoBoardSpec,
    calibrate_intrinsics,
    coverage_fraction,
    coverage_grid,
    detect_charuco,
    estimate_usable_radius,
    radial_fill,
    sharpness_score,
)
from umi.common.utility.print_utils import print_green, print_red, print_yellow

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
UNDISTORT_SUFFIX = "_undistort"


def add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--squares-x", type=int, default=14, help="Squares along X.")
    parser.add_argument("--squares-y", type=int, default=9, help="Squares along Y.")
    parser.add_argument(
        "--square-length", type=float, default=0.04, help="Square side, meters."
    )
    parser.add_argument(
        "--marker-length", type=float, default=0.03, help="Marker side, meters."
    )
    parser.add_argument(
        "--aruco-dict", type=str, default="DICT_5X5_1000", help="ArUco dictionary name."
    )
    parser.add_argument(
        "--legacy-pattern",
        action="store_true",
        help="Board was printed with the pre-OpenCV-4.6 marker parity.",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=10,
        help=(
            "Minimum ChArUco corners for a view to count. Keep this low: the "
            "views that cover the frame edges are partial views with few "
            "corners, and a high threshold rejects exactly those, capping "
            "achievable coverage."
        ),
    )


def board_spec_from_args(args) -> CharucoBoardSpec:
    return CharucoBoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict=args.aruco_dict,
        legacy_pattern=args.legacy_pattern,
    )


def view_signature(detection) -> np.ndarray:
    """Cheap pose proxy: corner centroid plus spread, in pixels.

    Used to reject near-duplicate views without running a full pose solve.
    """
    pts = detection.charuco_corners.reshape(-1, 2)
    centroid = pts.mean(axis=0)
    spread = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1).mean()))
    return np.array([centroid[0], centroid[1], spread], dtype=np.float64)


def is_novel(signature, accepted_signatures, novelty_px: float) -> bool:
    if not accepted_signatures:
        return True
    dists = [float(np.linalg.norm(signature - s)) for s in accepted_signatures]
    return min(dists) > novelty_px


def adds_new_cells(detection, mask, image_size, grid) -> bool:
    """True if this view lights up a grid cell no accepted view has reached.

    Checked before the novelty test: a view that fills an empty corner cell is
    valuable even when its centroid sits close to an already-accepted view.
    """
    cols, rows = int(grid[0]), int(grid[1])
    width, height = float(image_size[0]), float(image_size[1])
    pts = detection.charuco_corners.reshape(-1, 2)
    cx = np.clip((pts[:, 0] / width * cols).astype(int), 0, cols - 1)
    cy = np.clip((pts[:, 1] / height * rows).astype(int), 0, rows - 1)
    return bool((~mask[cy, cx]).any())


def read_factory_calibration(device, socket_name: str) -> dict:
    """Snapshot the on-EEPROM K/D/lens-model so it can be compared later.

    Best-effort: returns {} when the device has no factory block for the socket.
    """
    try:
        import depthai as dai

        from umi.common.hardware_sdk.camera.oak_device import _parse_socket

        socket = _parse_socket(socket_name)
        calib = device.device.readFactoryCalibration()
        M, width, height = calib.getDefaultIntrinsics(socket)
        out = {
            "K": np.asarray(M, dtype=np.float64).reshape(3, 3).tolist(),
            "native_resolution": [int(width), int(height)],
        }
        try:
            out["D"] = list(calib.getDistortionCoefficients(socket))
            model = calib.getDistortionModel(socket)
            out["distortion_model"] = str(model)
            out["is_fisheye"] = model == dai.CameraModel.Fisheye
        except Exception as e:
            out["distortion_error"] = str(e)
        return out
    except Exception as e:
        print_yellow(f"[intrinsics] Could not read factory calibration: {e}")
        return {}


def draw_overlay(image, detection, mask, accepted, target, sharp, sharp_min):
    vis = image.copy()
    h, w = vis.shape[:2]
    rows, cols = mask.shape
    # Shade grid cells that no accepted view has covered yet.
    for r in range(rows):
        for c in range(cols):
            if mask[r, c]:
                continue
            x0, y0 = int(c * w / cols), int(r * h / rows)
            x1, y1 = int((c + 1) * w / cols), int((r + 1) * h / rows)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (60, 60, 60), 1)
    if detection is not None:
        for pt in detection.charuco_corners.reshape(-1, 2):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
    n_corners = 0 if detection is None else detection.num_corners
    lines = [
        f"accepted {accepted}/{target}",
        f"corners {n_corners}",
        f"sharp {sharp:.0f} (min {sharp_min:.0f})",
        f"coverage {float(mask.mean()) * 100:.0f}%",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            vis, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
        )
        cv2.putText(
            vis, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )
    return vis


def cmd_capture(args) -> int:
    from umi.common.hardware_sdk.camera.oak_device import OAKDevice

    spec = board_spec_from_args(args)
    board = spec.build()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = args.resolution
    device = OAKDevice(oak_device_name=args.oak_device_name, fps=args.fps)
    device.build_pipeline(
        [
            {
                "name": "calib",
                "socket": args.socket,
                "resolution": [width, height],
            }
        ]
    )

    factory = read_factory_calibration(device, args.socket)
    device.start()
    stream = device.get_streams()[0]

    accepted_signatures = []
    detections = []
    sharpness_seen = []
    saved = 0
    grid = (args.grid_cols, args.grid_rows)

    try:
        # Discard early frames so auto-exposure/white-balance settle.
        for _ in range(args.warmup_frames):
            stream.get_frame()

        print_green(
            f"[intrinsics] Capturing. SPACE=force capture, q=finish. "
            f"Target {args.target_views} views."
        )
        while saved < args.target_views:
            frame = stream.get_frame()
            if frame is None or frame.rgb is None:
                print_yellow("[intrinsics] No frame (timeout); retrying.")
                continue
            image = frame.rgb
            detection = detect_charuco(
                image, board, min_corners=args.min_corners, source=f"live_{saved:03d}"
            )
            sharp = sharpness_score(image)
            mask = coverage_grid(detections, (width, height), grid)

            key = -1
            if not args.no_display:
                vis = draw_overlay(
                    image,
                    detection,
                    mask,
                    saved,
                    args.target_views,
                    sharp,
                    args.min_sharpness,
                )
                cv2.imshow("charuco intrinsics capture", vis)
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            forced = key == ord(" ")
            if detection is None:
                continue
            sharpness_seen.append(sharp)
            if not forced:
                # Absolute sharpness has no meaningful scale across cameras and
                # scenes, so gate relative to what this session is achieving.
                # Blurry views bias the focal length even when they detect fine.
                floor = args.min_sharpness
                if len(sharpness_seen) >= 8:
                    floor = max(
                        floor, args.sharpness_rel * float(np.median(sharpness_seen))
                    )
                if sharp < floor:
                    continue
                # Coverage wins over novelty: a view reaching an untouched cell
                # is always worth keeping, however similar its centroid.
                if not adds_new_cells(detection, mask, (width, height), grid):
                    if not is_novel(
                        view_signature(detection), accepted_signatures, args.novelty_px
                    ):
                        continue

            path = output_dir / f"view_{saved:03d}.png"
            cv2.imwrite(str(path), image)
            detection.source = path.name
            detections.append(detection)
            accepted_signatures.append(view_signature(detection))
            saved += 1
            print_green(
                f"[intrinsics] saved {path.name} "
                f"corners={detection.num_corners} sharp={sharp:.0f} "
                f"coverage={coverage_fraction(detections, (width, height), grid) * 100:.0f}%"
            )
    except KeyboardInterrupt:
        print_yellow("\n[intrinsics] Interrupted.")
    finally:
        device.stop()
        cv2.destroyAllWindows()

    meta = {
        "oak_device_name": args.oak_device_name,
        "socket": args.socket,
        "resolution": [width, height],
        "board": spec.to_dict(),
        "factory_calibration": factory,
        "num_views": saved,
    }
    (output_dir / "capture_meta.json").write_text(json.dumps(meta, indent=2))

    cov = coverage_fraction(detections, (width, height), grid)
    mask = coverage_grid(detections, (width, height), grid)
    print_green(f"[intrinsics] Saved {saved} views to {output_dir}")
    print_green(f"[intrinsics] Final coverage: {cov * 100:.0f}%")
    print_green(
        f"[intrinsics] Radial fill: "
        f"{radial_fill(detections, (width, height)) * 100:.0f}%"
    )
    if cov < 0.7:
        # Edge cells are the ones that constrain distortion, so name them
        # explicitly rather than reporting a single percentage.
        edge_missing = int((~mask[[0, -1], :]).sum() + (~mask[1:-1, [0, -1]]).sum())
        print_yellow(
            f"[intrinsics] Coverage {cov * 100:.0f}% — {edge_missing} border cells "
            f"still empty. Distortion at the frame edges is extrapolated, not "
            f"measured. Tilt the board steeply and let it run PARTLY OUT of "
            f"frame so corners land in the image corners; partial views are "
            f"wanted here, not a defect."
        )
    return 0


def cmd_calibrate(args) -> int:
    spec = board_spec_from_args(args)
    board = spec.build()
    image_dir = Path(args.image_dir)
    paths = sorted(
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
        # The undistort preview is written next to the inputs; re-running would
        # otherwise ingest it as a (double-width) capture.
        and UNDISTORT_SUFFIX not in p.stem
    )
    if not paths:
        print_red(f"[intrinsics] No images found in {image_dir}")
        return 1

    detections = []
    image_size = None
    max_intensity = None
    sharpness_seen = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print_yellow(f"[intrinsics] Unreadable: {path.name}")
            continue
        h, w = image.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            print_red(
                f"[intrinsics] Size mismatch: {path.name} is {w}x{h}, "
                f"expected {image_size[0]}x{image_size[1]}. "
                f"All images must come from one stream at one resolution."
            )
            return 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        max_intensity = (
            gray if max_intensity is None else np.maximum(max_intensity, gray)
        )
        sharpness_seen.append(sharpness_score(gray))
        detection = detect_charuco(
            image, board, min_corners=args.min_corners, source=path.name
        )
        if detection is None:
            print_yellow(f"[intrinsics] No board: {path.name}")
            continue
        if sharpness_seen[-1] < args.min_sharpness:
            print_yellow(f"[intrinsics] Too blurry, skipping: {path.name}")
            continue
        detections.append(detection)

    print_green(f"[intrinsics] {len(detections)}/{len(paths)} images usable.")
    if sharpness_seen:
        sv = np.array(sharpness_seen)
        print_green(
            f"[intrinsics] Sharpness: median={np.median(sv):.0f} "
            f"p10={np.percentile(sv, 10):.0f} min={sv.min():.0f}"
        )
        if sv.min() < 0.5 * np.median(sv):
            print_yellow(
                f"[intrinsics] Some views are far blurrier than the median. Blur "
                f"biases focal length; re-run with --min-sharpness "
                f"{0.6 * np.median(sv):.0f} to drop them and compare."
            )
    cov = coverage_fraction(detections, image_size, (args.grid_cols, args.grid_rows))
    print_green(f"[intrinsics] Corner coverage: {cov * 100:.0f}%")
    # Measure the lens image circle so radial fill is scored against what this
    # lens can actually image, not against unreachable frame corners.
    corner_r = float(np.hypot((image_size[0] - 1) / 2.0, (image_size[1] - 1) / 2.0))
    usable_r = estimate_usable_radius(max_intensity)
    reached_r = radial_fill(detections, image_size) * corner_r
    fill = reached_r / usable_r if usable_r > 0 else 0.0
    if usable_r < corner_r * 0.98:
        print_green(
            f"[intrinsics] Lens image circle ends at r={usable_r:.0f}px of a "
            f"{corner_r:.0f}px frame-corner radius — the frame corners are "
            f"outside this lens and can never be filled."
        )
    print_green(
        f"[intrinsics] Radial fill: {fill * 100:.0f}% of the usable radius "
        f"(corners reach r={reached_r:.0f}px of {usable_r:.0f}px)"
    )
    if fill < 0.9:
        print_yellow(
            f"[intrinsics] The outer {(1 - fill) * 100:.0f}% of the USABLE radius has "
            f"no corners in it, so distortion is extrapolated there. Work the "
            f"board out to r~{usable_r:.0f}px — the widest points of the image "
            f"circle, not the black frame corners."
        )

    try:
        result = calibrate_intrinsics(
            detections,
            board,
            image_size,
            model=args.model,
            min_views=args.min_views,
            board_spec=spec,
            fix_k3_k4=args.fix_k3_k4,
        )
    except ValueError as e:
        print_red(f"[intrinsics] Calibration failed: {e}")
        return 1
    except cv2.error as e:
        print_red(f"[intrinsics] OpenCV solver failed: {e}")
        return 1

    K, D = result.K, result.D
    if result.dropped_views:
        print_yellow(
            f"[intrinsics] Dropped {len(result.dropped_views)} view(s) that abort "
            f"fisheye initialisation (corners at extreme radius): "
            f"{', '.join(result.dropped_views)}"
        )
    print_green(f"[intrinsics] model={result.model} views={result.num_views}")
    print_green(f"[intrinsics] RMS reprojection error: {result.rms:.4f} px")
    print_green(
        f"[intrinsics] fx={K[0, 0]:.3f} fy={K[1, 1]:.3f} "
        f"cx={K[0, 2]:.3f} cy={K[1, 2]:.3f}"
    )
    print_green(f"[intrinsics] D={np.array2string(D, precision=6)}")

    worst = np.argsort(result.per_view_errors)[::-1][:5]
    print_green("[intrinsics] Worst views by reprojection error:")
    for i in worst:
        print(f"    {result.sources[i]}: {result.per_view_errors[i]:.4f} px")
    # Huge coefficients alongside a good RMS mean the model is trading
    # conditioning for fit — the extra terms are absorbing noise, not lens shape.
    if np.abs(D).max() > 100.0:
        print_yellow(
            f"[intrinsics] Distortion coefficients are very large "
            f"(max |D| = {np.abs(D).max():.1f}) despite RMS {result.rms:.3f} px. "
            f"The fit is ill-conditioned; treat this calibration as suspect and "
            f"check the undistort preview before using it."
        )
    if result.rms > args.max_rms:
        print_yellow(
            f"[intrinsics] RMS {result.rms:.3f} px exceeds --max-rms {args.max_rms}. "
            f"Consider dropping the worst views and re-running."
        )

    output = Path(args.output) if args.output else image_dir / "intrinsics.json"
    payload = result.to_dict()
    meta_path = image_dir / "capture_meta.json"
    if meta_path.exists():
        payload["capture_meta"] = json.loads(meta_path.read_text())
    output.write_text(json.dumps(payload, indent=2))
    print_green(f"[intrinsics] Wrote {output}")

    if args.undistort_preview:
        if result.model == "fisheye":
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D.reshape(4, 1), image_size, np.eye(3), balance=args.balance
            )
            # Reject near-zero focal lengths caused by an unstable distortion solve.
            # A wide lens with high --balance can legitimately reduce the focal length,
            # so do not reject a reduction that remains physically usable.
            ratio = float(new_K[0, 0]) / float(K[0, 0])
            if not 0.05 < ratio < 5.0:
                print_red(
                    f"[intrinsics] Undistort matrix is degenerate "
                    f"(new fx / fx = {ratio:.5f}). The high-order fisheye terms "
                    f"are overfit — usually too little coverage at the frame "
                    f"edges. Re-run with --fix-k3-k4, or recapture with corners "
                    f"reaching the image corners."
                )
                return 1
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D.reshape(4, 1), np.eye(3), new_K, image_size, cv2.CV_16SC2
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K, D, image_size, args.balance, image_size
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2
            )
        # The maps depend only on K/D, so build once and reuse for every sample.
        n = max(1, min(args.preview_count, len(detections)))
        picks = np.linspace(0, len(detections) - 1, n).round().astype(int)
        written = []
        for i, vi in enumerate(dict.fromkeys(picks.tolist())):
            src = detections[vi].source
            sample = cv2.imread(str(image_dir / src))
            if sample is None:
                continue
            undistorted = cv2.remap(
                sample,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            preview = np.hstack([sample, undistorted])
            stem = output.stem + UNDISTORT_SUFFIX
            if n > 1:
                stem = f"{stem}_{i:02d}_{Path(src).stem}"
            preview_path = output.with_name(stem + ".png")
            cv2.imwrite(str(preview_path), preview)
            written.append(preview_path.name)
        print_green(
            f"[intrinsics] Wrote {len(written)} preview(s) to {output.parent} "
            f"(left=raw, right=undistorted): {', '.join(written)}"
        )
        print_green(
            "[intrinsics] Check that straight edges in the scene are straight on "
            "the right. If they bow, the fit is bad regardless of RMS."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Capture board views from a live OAK stream.")
    add_board_args(cap)
    cap.add_argument(
        "--oak-device-name", type=str, default=None, help="e.g. oak-FFC-05"
    )
    cap.add_argument("--socket", type=str, default="CAM_A", help="CAM_A/CAM_B/CAM_C.")
    cap.add_argument(
        "--resolution", type=int, nargs=2, default=[640, 400], metavar=("W", "H")
    )
    cap.add_argument("--fps", type=int, default=30)
    cap.add_argument("--output-dir", type=str, required=True)
    cap.add_argument("--target-views", type=int, default=60)
    cap.add_argument("--warmup-frames", type=int, default=15)
    cap.add_argument(
        "--min-sharpness",
        type=float,
        default=60.0,
        help="Variance-of-Laplacian floor; rejects motion blur.",
    )
    cap.add_argument(
        "--sharpness-rel",
        type=float,
        default=0.6,
        help=(
            "Reject views below this fraction of the running median sharpness. "
            "Absolute thresholds do not transfer between cameras/scenes; this "
            "adapts. Set 0 to disable and use --min-sharpness alone."
        ),
    )
    cap.add_argument(
        "--novelty-px",
        type=float,
        default=25.0,
        help="Minimum view-signature distance from every accepted view.",
    )
    cap.add_argument("--grid-cols", type=int, default=8)
    cap.add_argument("--grid-rows", type=int, default=6)
    cap.add_argument("--no-display", action="store_true", help="Headless capture.")
    cap.set_defaults(func=cmd_capture)

    cal = sub.add_parser("calibrate", help="Solve intrinsics from a directory.")
    add_board_args(cal)
    cal.add_argument("--image-dir", type=str, required=True)
    cal.add_argument("--model", choices=["fisheye", "pinhole"], default="fisheye")
    cal.add_argument("--min-views", type=int, default=6)
    cal.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Drop images below this variance-of-Laplacian before solving.",
    )
    cal.add_argument("--output", type=str, default=None)
    cal.add_argument("--grid-cols", type=int, default=8)
    cal.add_argument("--grid-rows", type=int, default=6)
    cal.add_argument(
        "--max-rms", type=float, default=1.0, help="Warn above this RMS, in px."
    )
    cal.add_argument(
        "--fix-k3-k4",
        action="store_true",
        help=(
            "Fisheye only: pin k3/k4 to zero. Use when edge coverage is thin "
            "and the high-order terms overfit."
        ),
    )
    cal.add_argument(
        "--preview-count",
        type=int,
        default=1,
        help="How many undistort previews to write, sampled across the views.",
    )
    cal.add_argument(
        "--undistort-preview",
        action="store_true",
        help="Write a raw-vs-undistorted side-by-side image.",
    )
    cal.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="0=crop to valid pixels, 1=keep all pixels (black edges).",
    )
    cal.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
