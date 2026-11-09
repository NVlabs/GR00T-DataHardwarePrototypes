# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test whether T_AB = T_WA^{-1} @ T_WB is invariant under world coordinate axis swaps.

Convention: T_WA and T_WB both transform TO the world frame.
- T_WA: transforms points from frame A to World
- T_WB: transforms points from frame B to World
- T_AB = T_WA^{-1} @ T_WB: transforms points from B to A

When the world coordinate system changes (e.g., axes are swapped):
- If S is the axis swap transformation (from new world W' to old world W)
- T_W'A = S^{-1} @ T_WA  (A to W')
- T_W'B = S^{-1} @ T_WB  (B to W')
- T_AB' = (T_W'A)^{-1} @ T_W'B = (S^{-1} @ T_WA)^{-1} @ (S^{-1} @ T_WB)
       = T_WA^{-1} @ S @ S^{-1} @ T_WB = T_WA^{-1} @ T_WB = T_AB

So T_AB should remain unchanged regardless of the world frame used.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from umi.common.utility.pose import (
    convert_homogeneous_matrix,
    invert_transformation,
)


def create_random_transformation():
    """Create a random homogeneous transformation matrix."""
    # Random rotation
    rotation = R.random()
    rotation_matrix = rotation.as_matrix()

    # Random translation
    translation = np.random.uniform(-5, 5, size=3)

    return convert_homogeneous_matrix(rotation_matrix, translation)


def create_axis_swap_matrix(swap_type: str):
    """
    Create an axis swap transformation matrix.

    Args:
        swap_type: Type of axis swap, one of:
            - 'xy': swap X and Y axes
            - 'xz': swap X and Z axes
            - 'yz': swap Y and Z axes
            - 'xyz_to_zxy': rotate axes (X->Z, Y->X, Z->Y)
            - 'flip_x': flip X axis
            - 'flip_y': flip Y axis
            - 'flip_z': flip Z axis

    Returns:
        4x4 homogeneous transformation matrix for the axis swap
    """
    if swap_type == "xy":
        # Swap X and Y axes
        rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    elif swap_type == "xz":
        # Swap X and Z axes
        rot = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    elif swap_type == "yz":
        # Swap Y and Z axes
        rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    elif swap_type == "xyz_to_zxy":
        # Cyclic permutation: X->Z, Y->X, Z->Y
        rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    elif swap_type == "flip_x":
        # Flip X axis (mirror)
        rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elif swap_type == "flip_y":
        # Flip Y axis (mirror)
        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    elif swap_type == "flip_z":
        # Flip Z axis (mirror)
        rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    else:
        raise ValueError(f"Unknown swap type: {swap_type}")

    # No translation for pure axis swap
    return convert_homogeneous_matrix(rot, np.zeros(3))


def transform_to_new_world(T_WX, S):
    """
    Transform T_WX (X to World) to T_W'X (X to new World').

    If T_WX transforms from frame X to world W,
    and S is the transformation from new world W' to old world W,
    then T_W'X = S^{-1} @ T_WX (transforms from X to W')

    Args:
        T_WX: Transformation matrix from some frame X to old world W
        S: Axis swap matrix (from new world W' to old world W)

    Returns:
        Transformation matrix from X to new world W'
    """
    return invert_transformation(S) @ T_WX


@pytest.mark.parametrize(
    "swap_type",
    ["xy", "xz", "yz", "xyz_to_zxy", "flip_x", "flip_y", "flip_z"],
)
def test_transformation_invariance(
    swap_type: str, num_tests: int = 10, verbose: bool = True
):
    """
    Test that T_AB = T_WA^{-1} @ T_WB is invariant under world axis swap.

    Args:
        swap_type: Type of axis swap to test
        num_tests: Number of random tests to run
        verbose: If True, print all matrix multiplication details

    Returns:
        True if all tests pass, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"Testing axis swap type: '{swap_type}'")
    print("=" * 60)

    S = create_axis_swap_matrix(swap_type)
    S_inv = invert_transformation(S)
    print(f"\nAxis swap matrix S (W' -> W):\n{S}")
    print(f"\nS^{{-1}} (W -> W'):\n{S_inv}")

    all_passed = True

    for i in range(num_tests):
        print(f"\n{'-' * 60}")
        print(f"Test {i + 1}")
        print("-" * 60)

        # Create random transformations (both TO world)
        T_WA = create_random_transformation()  # A to World
        T_WB = create_random_transformation()  # B to World

        if verbose:
            print(f"\nT_WA (A -> World):\n{T_WA}")
            print(f"\nT_WB (B -> World):\n{T_WB}")

        # Compute T_AB in original world: T_AB = T_WA^{-1} @ T_WB
        T_WA_inv = invert_transformation(T_WA)
        T_AB_original = T_WA_inv @ T_WB

        if verbose:
            print("\n--- Original world: T_AB = T_WA^{-1} @ T_WB ---")
            print(f"{T_WA_inv}")
            print("        @")
            print(f"{T_WB}")
            print("        =")
            print(f"{T_AB_original}")

        # Transform both to new world coordinates
        T_WA_prime = transform_to_new_world(T_WA, S)  # A to W'
        T_WB_prime = transform_to_new_world(T_WB, S)  # B to W'

        if verbose:
            print("\n--- Transform to new world: T_W'X = S^{-1} @ T_WX ---")
            print("T_W'A = S^{-1} @ T_WA:")
            print(f"{S_inv}")
            print("        @")
            print(f"{T_WA}")
            print("        =")
            print(f"{T_WA_prime}")

            print("\nT_W'B = S^{-1} @ T_WB:")
            print(f"{S_inv}")
            print("        @")
            print(f"{T_WB}")
            print("        =")
            print(f"{T_WB_prime}")

        # Compute T_AB in new world: T_AB' = T_W'A^{-1} @ T_W'B
        T_WA_prime_inv = invert_transformation(T_WA_prime)
        T_AB_new = T_WA_prime_inv @ T_WB_prime

        if verbose:
            print("\n--- New world: T_AB' = (T_W'A)^{-1} @ T_W'B ---")
            print(f"{T_WA_prime_inv}")
            print("        @")
            print(f"{T_WB_prime}")
            print("        =")
            print(f"{T_AB_new}")

        # Check if they are equal
        diff = np.abs(T_AB_original - T_AB_new)
        max_diff = np.max(diff)

        if max_diff < 1e-10:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False

        print("\n--- Comparison ---")
        print(f"T_AB (original):\n{T_AB_original}")
        print(f"\nT_AB' (new world):\n{T_AB_new}")
        print(f"\nMax difference: {max_diff:.2e}")
        print(f"Result: {status}")

    print(f"\n{'=' * 60}")
    print(
        f"Result for '{swap_type}': {'ALL PASSED' if all_passed else 'SOME FAILED'} ({num_tests} tests)"
    )

    assert all_passed


def test_with_specific_example():
    """
    Test with a specific example for easy verification.
    """
    print("\n" + "=" * 60)
    print("SPECIFIC EXAMPLE TEST")
    print("=" * 60)

    # Create specific transformations (both TO world)
    # T_WA: Frame A to World (simple rotation around Z and translation)
    R_WA = R.from_euler("z", 45, degrees=True).as_matrix()
    t_WA = np.array([1, 2, 3])
    T_WA = convert_homogeneous_matrix(R_WA, t_WA)

    # T_WB: Frame B to World (rotation around X and translation)
    R_WB = R.from_euler("x", 30, degrees=True).as_matrix()
    t_WB = np.array([0.5, 1.5, 2.5])
    T_WB = convert_homogeneous_matrix(R_WB, t_WB)

    print("\n" + "-" * 60)
    print("Step 1: Original transformations (both TO world)")
    print("-" * 60)
    print(f"T_WA (A -> World):\n{T_WA}")
    print(f"\nT_WB (B -> World):\n{T_WB}")

    print("\n" + "-" * 60)
    print("Step 2: Compute T_AB in original world")
    print("-" * 60)
    T_WA_inv = invert_transformation(T_WA)
    print("T_AB = T_WA^{-1} @ T_WB")
    print("\nMatrix multiplication:")
    print(f"{T_WA_inv}")
    print("        @")
    print(f"{T_WB}")
    print("        =")
    T_AB_original = T_WA_inv @ T_WB
    print(f"{T_AB_original}")

    print("\n" + "-" * 60)
    print("Step 3: Define axis swap matrix S (XY swap)")
    print("-" * 60)
    S = create_axis_swap_matrix("xy")
    S_inv = invert_transformation(S)

    print("\n" + "-" * 60)
    print("Step 4: Transform T_WA and T_WB to new world")
    print("-" * 60)
    print("T_W'A = S^{-1} @ T_WA")
    print("\nMatrix multiplication:")
    print(f"{S_inv}")
    print("        @")
    print(f"{T_WA}")
    print("        =")
    T_WA_prime = S_inv @ T_WA
    print(f"{T_WA_prime}")

    print("\n" + "-" * 40)
    print("T_W'B = S^{-1} @ T_WB")
    print("\nMatrix multiplication:")
    print(f"{S_inv}")
    print("        @")
    print(f"{T_WB}")
    print("        =")
    T_WB_prime = S_inv @ T_WB
    print(f"{T_WB_prime}")

    print("\n" + "-" * 60)
    print("Step 5: Compute T_AB in new world")
    print("-" * 60)
    T_WA_prime_inv = invert_transformation(T_WA_prime)
    print("(T_W'A)^{-1} @ T_W'B")
    print("\nMatrix multiplication:")
    print(f"{T_WA_prime_inv}")
    print("        @")
    print(f"{T_WB_prime}")
    print("        =")
    T_AB_new = T_WA_prime_inv @ T_WB_prime
    print(f"{T_AB_new}")

    print("\n" + "-" * 60)
    print("Step 6: Verify mathematical identity")
    print("-" * 60)
    print("T_AB' = (S^{-1} @ T_WA)^{-1} @ (S^{-1} @ T_WB)")
    print("      = T_WA^{-1} @ S @ S^{-1} @ T_WB")
    print("      = T_WA^{-1} @ I @ T_WB")
    print("      = T_WA^{-1} @ T_WB")
    print("      = T_AB")
    print("\nVerifying S @ S^{-1} = I:")
    print(f"{S}")
    print("        @")
    print(f"{S_inv}")
    print("        =")
    S_times_S_inv = S @ S_inv
    print(f"{S_times_S_inv}")

    print("\n" + "-" * 60)
    print("Step 7: Compare results")
    print("-" * 60)
    print(f"T_AB (original):\n{T_AB_original}")
    print(f"\nT_AB' (new world):\n{T_AB_new}")
    diff = np.max(np.abs(T_AB_original - T_AB_new))
    print(f"\nMax difference: {diff:.2e}")
    print(f"Result: {'PASS' if diff < 1e-10 else 'FAIL'}")

    assert diff < 1e-10


def test_point_transformation():
    """
    Verify the transformation using actual point transformations.
    This provides geometric intuition for why T_AB is invariant.
    """
    print("\n" + "=" * 60)
    print("POINT TRANSFORMATION TEST")
    print("=" * 60)

    # Create random transformations (both TO world)
    T_WA = create_random_transformation()  # A to World
    T_WB = create_random_transformation()  # B to World

    # A point in frame B
    p_B = np.array([1, 2, 3, 1])  # Homogeneous coordinates

    # Transform through original world
    p_W = T_WB @ p_B  # B -> World
    p_A = invert_transformation(T_WA) @ p_W  # World -> A

    print(f"Point in B: {p_B[:3]}")
    print(f"Point in W (via T_WB): {p_W[:3]}")
    print(f"Point in A (via T_WA^-1): {p_A[:3]}")

    # Direct transformation
    T_AB = invert_transformation(T_WA) @ T_WB
    p_A_direct = T_AB @ p_B
    print(f"Point in A (via T_AB): {p_A_direct[:3]}")

    # Repeat with the world frames exchanged.
    S = create_axis_swap_matrix("xz")
    T_WA_prime = transform_to_new_world(T_WA, S)
    T_WB_prime = transform_to_new_world(T_WB, S)

    p_W_prime = T_WB_prime @ p_B  # B -> World'
    p_A_new = invert_transformation(T_WA_prime) @ p_W_prime  # World' -> A

    print("\nWith XZ swap:")
    print(f"Point in W' (via T_W'B): {p_W_prime[:3]}")
    print(f"Point in A (via T_W'A^-1): {p_A_new[:3]}")

    # Via new T_AB
    T_AB_new = invert_transformation(T_WA_prime) @ T_WB_prime
    p_A_new_direct = T_AB_new @ p_B
    print(f"Point in A (via T_AB'): {p_A_new_direct[:3]}")

    diff = np.max(np.abs(p_A[:3] - p_A_new[:3]))
    print(f"\nMax difference in final point: {diff:.2e}")
    print(f"Result: {'PASS' if diff < 1e-10 else 'FAIL'}")

    assert diff < 1e-10


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(42)  # For reproducibility

    print("\n" + "#" * 60)
    print("# Testing T_AB invariance under world coordinate axis swaps")
    print("# Convention: T_WA, T_WB both transform TO world")
    print("# T_AB = T_WA^{-1} @ T_WB")
    print("#" * 60)

    # Run specific example first
    test_with_specific_example()
