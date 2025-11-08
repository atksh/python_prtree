#!/usr/bin/env python3
"""
Extreme test cases for float32 precision issues.

This test attempts to find actual false negatives by testing
extreme precision boundaries.
"""
import numpy as np
from python_prtree import PRTree2D


def create_barely_overlapping_boxes():
    """
    Create boxes that overlap by the smallest possible float64 amount.
    """
    print("\n=== Extreme Test: Minimal Overlap ===")

    # Start with a clean float32 value
    base = np.float32(100.0)

    # Get the next representable float32 value
    next_val = np.nextafter(base, np.float32(np.inf))
    ulp = next_val - base

    print(f"Base value (float32): {base}")
    print(f"Next value (float32): {next_val}")
    print(f"ULP: {ulp}")

    # In float64, create boxes that overlap by less than the float32 ULP
    # Box A: ends at base + (ulp/3) in float64
    # Box B: starts at base in float64

    # When converted to float32:
    # Box A max will round to base (losing the tiny extension)
    # Box B min is exactly base
    # They should touch at base in float32

    box_a_end_f64 = np.float64(base) + np.float64(ulp) / 3.0
    box_b_start_f64 = np.float64(base)

    print(f"\nBox A end (float64): {box_a_end_f64:.20f}")
    print(f"Box B start (float64): {box_b_start_f64:.20f}")
    print(f"Overlap in float64: {box_a_end_f64 >= box_b_start_f64}")

    box_a_end_f32 = np.float32(box_a_end_f64)
    box_b_start_f32 = np.float32(box_b_start_f64)

    print(f"\nBox A end (float32): {box_a_end_f32:.20f}")
    print(f"Box B start (float32): {box_b_start_f32:.20f}")
    print(f"Overlap in float32: {box_a_end_f32 >= box_b_start_f32}")

    # Build trees with both precisions
    boxes_f64 = np.array([
        [0.0, 0.0, box_a_end_f64, 100.0],
        [box_b_start_f64, 0.0, 200.0, 100.0],
    ], dtype=np.float64)

    boxes_f32 = np.array([
        [0.0, 0.0, box_a_end_f32, 100.0],
        [box_b_start_f32, 0.0, 200.0, 100.0],
    ], dtype=np.float32)

    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree intersections: {pairs_f64}")
    print(f"Float32 tree intersections: {pairs_f32}")

    if len(pairs_f64) != len(pairs_f32):
        print(f"\n❌ FALSE NEGATIVE DETECTED!")
        return True

    return False


def test_negative_coordinates():
    """
    Test with negative coordinates where float32 precision differs.
    """
    print("\n=== Test: Negative Coordinates ===")

    # Float32 precision varies with magnitude
    # Test negative values
    val = -16777216.5  # Near 2^24 boundary

    boxes_f64 = np.array([
        [val - 1000, 0.0, val + 0.1, 100.0],
        [val, 0.0, val + 1000, 100.0],
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"Boxes (float64):\n{boxes_f64}")
    print(f"Boxes (float32):\n{boxes_f32}")

    # Check if the small overlap is preserved
    overlap_f64 = boxes_f64[0, 2] >= boxes_f64[1, 0]
    overlap_f32 = boxes_f32[0, 2] >= boxes_f32[1, 0]

    print(f"\nOverlap in float64: {overlap_f64}")
    print(f"Overlap in float32: {overlap_f32}")

    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree intersections: {pairs_f64}")
    print(f"Float32 tree intersections: {pairs_f32}")

    if overlap_f64 and not overlap_f32:
        print(f"\n⚠️  Overlap lost in float32!")

    if len(pairs_f64) != len(pairs_f32):
        print(f"\n❌ FALSE NEGATIVE DETECTED!")
        return True

    return False


def test_accumulated_rounding_errors():
    """
    Test if accumulated rounding errors can cause false negatives.
    """
    print("\n=== Test: Accumulated Rounding Errors ===")

    # Create a chain of boxes that should all touch
    # Each box ends where the next one begins
    # With float32 rounding, gaps might appear

    n_boxes = 10
    base = np.float64(0.1)  # Not exactly representable in binary

    boxes_f64 = []
    for i in range(n_boxes):
        start = base * i
        end = base * (i + 1)
        boxes_f64.append([start, 0.0, end, 100.0])

    boxes_f64 = np.array(boxes_f64, dtype=np.float64)
    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"Number of boxes: {n_boxes}")
    print(f"\nFirst 3 boxes (float64):\n{boxes_f64[:3]}")
    print(f"\nFirst 3 boxes (float32):\n{boxes_f32[:3]}")

    # Check for gaps
    for i in range(n_boxes - 1):
        gap_f64 = boxes_f64[i + 1, 0] - boxes_f64[i, 2]
        gap_f32 = boxes_f32[i + 1, 0] - boxes_f32[i, 2]

        if abs(gap_f32) > abs(gap_f64) + 1e-10:
            print(f"\nGap created between box {i} and {i+1}:")
            print(f"  Float64 gap: {gap_f64:.15e}")
            print(f"  Float32 gap: {gap_f32:.15e}")

    idx = np.arange(n_boxes, dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree intersections: {len(pairs_f64)} pairs")
    print(f"Float32 tree intersections: {len(pairs_f32)} pairs")

    if len(pairs_f64) != len(pairs_f32):
        print(f"\n❌ FALSE NEGATIVE DETECTED!")
        print(f"Missing pairs: {len(pairs_f64) - len(pairs_f32)}")
        return True

    return False


def test_subnormal_values():
    """
    Test with very small (subnormal) float32 values.
    """
    print("\n=== Test: Subnormal Float32 Values ===")

    # Smallest normal float32: ~1.175e-38
    # Smallest subnormal float32: ~1.4e-45

    small = np.float32(1e-40)
    tiny_overlap = np.float32(1e-44)

    boxes_f32 = np.array([
        [0.0, 0.0, small + tiny_overlap, 1.0],
        [small, 0.0, 1.0, 1.0],
    ], dtype=np.float32)

    print(f"Small value: {small}")
    print(f"Tiny overlap: {tiny_overlap}")
    print(f"Sum: {small + tiny_overlap}")
    print(f"\nBoxes (float32):\n{boxes_f32}")

    overlap = boxes_f32[0, 2] >= boxes_f32[1, 0]
    print(f"\nBoxes overlap: {overlap}")

    idx = np.array([0, 1], dtype=np.int64)

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"Intersections: {pairs_f32}")

    # These should intersect based on the values
    if overlap and len(pairs_f32) == 0:
        print(f"\n❌ FALSE NEGATIVE DETECTED with subnormal values!")
        return True

    return False


def test_double_rounding():
    """
    Test double rounding: float64 -> float32 -> comparison.
    """
    print("\n=== Test: Double Rounding Effect ===")

    # Create a value in float64 that has different rounding behavior
    # Specifically, a value that rounds differently when going directly
    # to float32 vs when being used in comparisons

    # Use a value that's exactly between two float32 representations
    f32_val = np.float32(1000.0)
    f32_next = np.nextafter(f32_val, np.float32(np.inf))
    f32_prev = np.nextafter(f32_val, np.float32(-np.inf))

    # Create a float64 value exactly between f32_val and f32_next
    midpoint_f64 = (np.float64(f32_val) + np.float64(f32_next)) / 2.0

    print(f"f32_prev: {f32_prev:.20f}")
    print(f"f32_val:  {f32_val:.20f}")
    print(f"midpoint (f64): {midpoint_f64:.20f}")
    print(f"f32_next: {f32_next:.20f}")

    # When converted to float32, midpoint rounds to...
    midpoint_f32 = np.float32(midpoint_f64)
    print(f"midpoint (f32): {midpoint_f32:.20f}")

    # Create boxes using this
    boxes_f64 = np.array([
        [0.0, 0.0, midpoint_f64, 100.0],
        [f32_val, 0.0, 2000.0, 100.0],
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nBoxes (float64):\n{boxes_f64}")
    print(f"Boxes (float32):\n{boxes_f32}")

    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree intersections: {pairs_f64}")
    print(f"Float32 tree intersections: {pairs_f32}")

    if len(pairs_f64) != len(pairs_f32):
        print(f"\n❌ FALSE NEGATIVE DETECTED!")
        return True

    return False


if __name__ == "__main__":
    print("Extreme float32 precision tests")
    print("=" * 70)

    issue_found = False

    issue_found |= create_barely_overlapping_boxes()
    issue_found |= test_negative_coordinates()
    issue_found |= test_accumulated_rounding_errors()
    issue_found |= test_subnormal_values()
    issue_found |= test_double_rounding()

    print("\n" + "=" * 70)
    if issue_found:
        print("❌ ISSUE CONFIRMED: Float32 rounding causes false negatives!")
    else:
        print("⚠️  No false negatives detected in extreme tests")
        print("The issue may require specific real-world data to reproduce")
    print("=" * 70)
