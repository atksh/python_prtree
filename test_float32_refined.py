#!/usr/bin/env python3
"""
Refined test for float32 rounding false negative in overlap detection.

Focus on cases where float32 rounding causes actual precision loss in the tree structure.
"""
import numpy as np
from python_prtree import PRTree2D


def test_float32_internal_representation():
    """
    Test the actual internal float32 representation issue.

    When boxes are stored as float32 internally but queried with float64 values,
    there can be mismatches due to rounding.
    """
    print("\n=== Test Case: Float32 Internal Representation ===")

    # Create two boxes that should overlap based on float64 values
    # but might not overlap when stored as float32

    # We need values that change when converted to float32
    # Example: 0.1 cannot be exactly represented in binary floating point

    val1_f64 = np.float64(0.1) * 1000  # 100.0 in float64
    val2_f64 = val1_f64 + np.float64(0.0000001)  # Slightly larger

    print(f"val1 (float64): {val1_f64:.20f}")
    print(f"val2 (float64): {val2_f64:.20f}")

    val1_f32 = np.float32(val1_f64)
    val2_f32 = np.float32(val2_f64)

    print(f"val1 (float32): {val1_f32:.20f}")
    print(f"val2 (float32): {val2_f32:.20f}")

    # Check if they're different after float32 conversion
    if val1_f32 == val2_f32:
        print(f"⚠️  Float32 rounding makes these values identical!")
        print(f"This can cause precision issues.")

    # Create boxes using these values
    boxes_f64 = np.array([
        [0.0, 0.0, val2_f64, 100.0],  # Box 0
        [val1_f64, 0.0, 200.0, 100.0], # Box 1
    ], dtype=np.float64)

    boxes_f32 = np.array([
        [0.0, 0.0, val2_f32, 100.0],  # Box 0
        [val1_f32, 0.0, 200.0, 100.0], # Box 1
    ], dtype=np.float32)

    print(f"\nBoxes (float64):\n{boxes_f64}")
    print(f"\nBoxes (float32):\n{boxes_f32}")

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


def test_float32_tree_float64_query_mismatch():
    """
    Test mismatch between float32 tree storage and float64 query.

    When a tree is built with float32, the bounding boxes are stored with
    float32 precision. If we then query with float64 values that represent
    the "true" boundaries, we might miss intersections due to rounding.
    """
    print("\n=== Test Case: Float32 Tree with Float64 Query Mismatch ===")

    # Create a value that has different representations in float32 vs float64
    # Use a value that's not exactly representable in binary
    true_val_f64 = np.float64(1e7) + np.float64(0.3)  # 10000000.3

    print(f"True value (float64): {true_val_f64:.10f}")

    # Convert to float32 and back to see the precision loss
    rounded_val_f32 = np.float32(true_val_f64)
    print(f"Rounded value (float32): {rounded_val_f32:.10f}")
    print(f"Difference: {true_val_f64 - np.float64(rounded_val_f32):.15f}")

    # Box A: ends at the rounded float32 value
    # Box B: starts at the true float64 value
    # These should touch/overlap in float64 but might have a gap in float32

    box_a_f32 = np.array([[0.0, 0.0, rounded_val_f32, 100.0]], dtype=np.float32)
    box_b_f32 = np.array([[rounded_val_f32, 0.0, rounded_val_f32 + 1000, 100.0]], dtype=np.float32)

    # Build tree with float32
    idx = np.array([0, 1], dtype=np.int64)
    boxes_f32 = np.vstack([box_a_f32, box_b_f32])
    tree_f32 = PRTree2D(idx, boxes_f32)

    print(f"\nTree boxes (float32):\n{boxes_f32}")

    # Query with the original float64 value
    query_at_true_boundary = np.array([true_val_f64, 0.0, true_val_f64 + 10, 100.0], dtype=np.float64)
    print(f"\nQuery box (float64): {query_at_true_boundary}")

    # This query should intersect box A (whose end is at the rounded value)
    # But due to float32 storage, it might not
    result = tree_f32.query(query_at_true_boundary.tolist())
    print(f"\nQuery result: {result}")

    # The query box is between the boxes, so it might not intersect either
    # Let's also test intersection detection
    pairs = tree_f32.query_intersections()
    print(f"Intersections: {pairs}")

    # Expected: boxes should touch at the boundary
    print(f"\nBox A max: {boxes_f32[0, 2]:.10f}")
    print(f"Box B min: {boxes_f32[1, 0]:.10f}")
    print(f"Boxes touch: {boxes_f32[0, 2] >= boxes_f32[1, 0]}")

    return False


def test_float32_gap_creation():
    """
    Test if float32 rounding can create gaps between boxes that should touch.
    """
    print("\n=== Test Case: Gap Creation via Float32 Rounding ===")

    # Find two float64 values that when rounded to float32, create a gap
    # Strategy: Find a float64 value v such that float32(v) < v < float32(nextafter(float32(v)))

    base = np.float64(16777216.0)  # 2^24, where float32 has integer precision
    delta = np.float64(1.5)  # This should round differently

    val1_f64 = base + delta
    val2_f64 = base + delta

    val1_f32 = np.float32(val1_f64)
    next_f32 = np.nextafter(val1_f32, np.float32(np.inf))

    print(f"Base + delta (float64): {val1_f64:.10f}")
    print(f"Rounded down (float32): {val1_f32:.10f}")
    print(f"Next float32 value: {next_f32:.10f}")
    print(f"Gap size: {np.float64(next_f32) - np.float64(val1_f32):.10f}")

    # Create boxes that touch at val1_f64 in float64
    # But when stored as float32, they might have a gap
    boxes_f64 = np.array([
        [0.0, 0.0, val1_f64, 100.0],  # Box A ends at val1_f64
        [val1_f64, 0.0, val1_f64 + 1000, 100.0],  # Box B starts at val1_f64
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nOriginal boxes (float64):\n{boxes_f64}")
    print(f"\nRounded boxes (float32):\n{boxes_f32}")

    # Check if gap was created
    gap_f64 = boxes_f64[1, 0] - boxes_f64[0, 2]
    gap_f32 = boxes_f32[1, 0] - boxes_f32[0, 2]

    print(f"\nGap in float64: {gap_f64:.15f}")
    print(f"Gap in float32: {gap_f32:.15f}")

    if gap_f32 > gap_f64:
        print(f"⚠️  Float32 rounding created a gap!")

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


def test_specific_failing_case():
    """
    Test a specific case that's known to fail with float32 precision.
    """
    print("\n=== Test Case: Specific Failing Case ===")

    # Use a known problematic value
    # At large magnitudes, float32 has ULP (Unit in Last Place) of several units
    large_val = 16777216.0  # 2^24

    # Create boxes that overlap by less than 1 ULP at this magnitude
    ulp = np.nextafter(np.float32(large_val), np.float32(np.inf)) - np.float32(large_val)
    print(f"ULP at {large_val}: {ulp}")

    # Create boxes that overlap by half a ULP in float64
    overlap = ulp / 2.0

    boxes_f64 = np.array([
        [0.0, 0.0, large_val + overlap, 100.0],
        [large_val, 0.0, large_val + 1000, 100.0],
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nBoxes (float64):\n{boxes_f64}")
    print(f"Boxes (float32):\n{boxes_f32}")

    # Check overlap
    overlap_f64 = boxes_f64[0, 2] >= boxes_f64[1, 0]
    overlap_f32 = boxes_f32[0, 2] >= boxes_f32[1, 0]

    print(f"\nOverlap in float64: {overlap_f64}")
    print(f"Overlap in float32: {overlap_f32}")

    if overlap_f64 and not overlap_f32:
        print(f"⚠️  Overlap lost in float32 conversion!")

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
    print("Refined float32 rounding false negative tests")
    print("=" * 70)

    issue_found = False

    issue_found |= test_float32_internal_representation()
    issue_found |= test_float32_tree_float64_query_mismatch()
    issue_found |= test_float32_gap_creation()
    issue_found |= test_specific_failing_case()

    print("\n" + "=" * 70)
    if issue_found:
        print("❌ ISSUE CONFIRMED: Float32 rounding causes false negatives")
    else:
        print("✅ No false negatives detected in refined tests")
    print("=" * 70)
