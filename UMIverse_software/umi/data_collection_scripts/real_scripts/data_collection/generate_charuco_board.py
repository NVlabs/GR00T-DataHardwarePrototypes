#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate the Charuco board image for camera pose calibration.

The defaults reproduce the board the station detects: the `marker` block of the
`camera.relative_calibration` check in config/data_recording/production/*.yaml
(14x9 squares, 40mm squares, 30mm markers, DICT_5X5_1000), at true size when
printed at 300 DPI.

Usage:
    uv run python -m real_scripts.data_collection.generate_charuco_board --output charuco_board.png
"""

import argparse

import cv2
import numpy as np

from umi.common.utility.print_utils import print_green


def main():
    parser = argparse.ArgumentParser(
        description="Generate Charuco board for camera pose calibration"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="charuco_board.png",
        help="Output path for generated Charuco board",
    )
    parser.add_argument(
        "--squares-x",
        type=int,
        default=14,
        help="Number of squares in X direction on Charuco board",
    )
    parser.add_argument(
        "--squares-y",
        type=int,
        default=9,
        help="Number of squares in Y direction on Charuco board",
    )
    parser.add_argument(
        "--square-length",
        type=float,
        default=0.04,
        help="Length of each square in meters",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.03,
        help="Length of ArUco marker in meters",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=6614,
        help="Image width in pixels (default: 6614 = 14 x 40mm at 300 DPI)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=4252,
        help="Image height in pixels (default: 4252 = 9 x 40mm at 300 DPI)",
    )
    parser.add_argument(
        "--marker-id-offset",
        type=int,
        default=0,
        help=(
            "Starting ArUco marker ID (default 0; use the same value in the "
            "detector config so the IDs on the printed board line up with "
            "what the detector expects)"
        ),
    )
    parser.add_argument(
        "--legacy-pattern",
        action="store_true",
        help=(
            "Emit the pre-OpenCV-4.6 ChArUco marker layout "
            "(match the detector's legacy_pattern setting)"
        ),
    )

    args = parser.parse_args()

    print_green("Generating Charuco board...")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    n_markers = (args.squares_x * args.squares_y) // 2
    ids = np.arange(
        args.marker_id_offset,
        args.marker_id_offset + n_markers,
        dtype=np.int32,
    )
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        aruco_dict,
        ids,
    )
    if args.legacy_pattern:
        board.setLegacyPattern(True)

    board_image = board.generateImage((args.width, args.height))
    cv2.imwrite(args.output, board_image)

    print_green(f"\nCharuco board saved to {args.output}")
    print("\nBoard specifications:")
    print(f"  - Squares: {args.squares_x}x{args.squares_y}")
    print(f"  - Square length: {args.square_length}m ({args.square_length * 100}cm)")
    print(f"  - Marker length: {args.marker_length}m ({args.marker_length * 100}cm)")
    print(f"  - Image size: {args.width}x{args.height} pixels")
    print(
        f"  - Marker IDs: {args.marker_id_offset}..{args.marker_id_offset + n_markers - 1}"
    )
    print(f"  - Legacy pattern: {args.legacy_pattern}")
    print("\nNext steps:")
    print("  1. Print this image at 300 DPI, 100% scale (no fit-to-page)")
    print("  2. Measure the actual square size with a ruler")
    print("  3. Put the measured square_length and marker_length in the marker")
    print("     block of the recording config's camera.relative_calibration check")


if __name__ == "__main__":
    main()
