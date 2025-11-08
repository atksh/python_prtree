#!/usr/bin/env python3
"""
Test to reproduce float32 rounding false negative issue in overlap detection.

Issue: When using float32 input, rounding errors can cause overlapping AABBs
to not be detected as intersecting (false negative).
"""
import numpy as np
from python_prtree import PRTree2D


def test_float32_false_negative():
    """
    Reproduce the false negative issue with float32 rounding.

    Create two boxes that overlap very slightly in float64 precision,
    but may not overlap after float32 rounding.
    """
    print("\n=== Test Case 1: Very small overlap ===")

    # Create boxes with very small overlap that might be lost in float32
    # Box A: [0.0, 100.00001]
    # Box B: [100.0, 200.0]
    # These should overlap at the boundary in float64, but may not in float32

    # Using float64
    boxes_f64 = np.array([
        [0.0, 0.0, 100.00001, 100.0],  # Box 0: slightly overlapping
        [100.0, 0.0, 200.0, 100.0],     # Box 1
    ], dtype=np.float64)

    # Using float32
    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nOriginal float64 boxes:\n{boxes_f64}")
    print(f"\nFloat32 boxes after conversion:\n{boxes_f32}")
    print(f"\nPrecision loss: {boxes_f64 - boxes_f32.astype(np.float64)}")

    idx = np.array([0, 1], dtype=np.int64)

    # Test with float64 input (should use refinement)
    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    # Test with float32 input (no refinement)
    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree intersections: {pairs_f64}")
    print(f"Float32 tree intersections: {pairs_f32}")

    if len(pairs_f64) != len(pairs_f32):
        print(f"\n⚠️  FALSE NEGATIVE DETECTED!")
        print(f"Float64 found {len(pairs_f64)} pairs, but float32 found {len(pairs_f32)} pairs")
        return True

    return False


def test_float32_precision_boundary():
    """
    Test with values at float32 precision boundary.

    Float32 has about 7 decimal digits of precision.
    """
    print("\n=== Test Case 2: Float32 precision boundary ===")

    # Create a case where float32 rounding causes non-overlap
    # Use large coordinates where float32 has limited precision
    base = 1000000.0  # 1 million
    epsilon = 0.00001  # Small overlap

    boxes_f64 = np.array([
        [0.0, 0.0, base + epsilon, 100.0],
        [base, 0.0, base + 100.0, 100.0],
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nOriginal float64 boxes:\n{boxes_f64}")
    print(f"\nFloat32 boxes after conversion:\n{boxes_f32}")

    # Check if the overlap is preserved
    box0_max_x_f64 = boxes_f64[0, 2]
    box1_min_x_f64 = boxes_f64[1, 0]
    box0_max_x_f32 = boxes_f32[0, 2]
    box1_min_x_f32 = boxes_f32[1, 0]

    print(f"\nFloat64: Box0 max_x={box0_max_x_f64:.10f}, Box1 min_x={box1_min_x_f64:.10f}")
    print(f"Float64 overlap: {box0_max_x_f64 >= box1_min_x_f64}")

    print(f"\nFloat32: Box0 max_x={box0_max_x_f32:.10f}, Box1 min_x={box1_min_x_f32:.10f}")
    print(f"Float32 overlap: {box0_max_x_f32 >= box1_min_x_f32}")

    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree intersections: {pairs_f64}")
    print(f"Float32 tree intersections: {pairs_f32}")

    if len(pairs_f64) != len(pairs_f32):
        print(f"\n⚠️  FALSE NEGATIVE DETECTED!")
        print(f"Float64 found {len(pairs_f64)} pairs, but float32 found {len(pairs_f32)} pairs")
        return True

    return False


def test_float32_query_method():
    """
    Test the query() method with float32 to check for false negatives.
    """
    print("\n=== Test Case 3: Query method with float32 ===")

    # Create a box and a query box that barely overlap
    boxes_f64 = np.array([
        [0.0, 0.0, 100.0000001, 100.0],
    ], dtype=np.float64)

    query_f64 = np.array([100.0, 0.0, 200.0, 100.0], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)
    query_f32 = query_f64.astype(np.float32)

    print(f"\nBox (float64): {boxes_f64[0]}")
    print(f"Box (float32): {boxes_f32[0]}")
    print(f"Query (float64): {query_f64}")
    print(f"Query (float32): {query_f32}")

    idx = np.array([0], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    result_f64 = tree_f64.query(query_f64)

    tree_f32 = PRTree2D(idx, boxes_f32)
    result_f32 = tree_f32.query(query_f32.tolist())

    print(f"\nFloat64 tree query result: {result_f64}")
    print(f"Float32 tree query result: {result_f32}")

    if len(result_f64) != len(result_f32):
        print(f"\n⚠️  FALSE NEGATIVE DETECTED in query()!")
        print(f"Float64 found {len(result_f64)} items, but float32 found {len(result_f32)} items")
        return True

    return False


def test_float32_ulp_case():
    """
    Test with boxes separated by exactly 1 ULP (Unit in the Last Place) in float32.
    """
    print("\n=== Test Case 4: 1 ULP separation in float32 ===")

    # In float32, around 100.0, the ULP is about 0.0000076
    # We'll create boxes that touch exactly at the float32 representation boundary

    val1 = np.float32(100.0)
    # Get the next representable float32 value
    val2 = np.nextafter(val1, np.float32(np.inf))

    print(f"\nval1 (float32): {val1:.20f}")
    print(f"val2 (nextafter): {val2:.20f}")
    print(f"Difference: {val2 - val1:.20e}")

    # Create boxes that should touch/overlap at the boundary
    boxes_f32 = np.array([
        [0.0, 0.0, val1, 100.0],
        [val1, 0.0, 200.0, 100.0],
    ], dtype=np.float32)

    # Convert to float64 for comparison
    boxes_f64 = boxes_f32.astype(np.float64)

    print(f"\nBoxes (float32):\n{boxes_f32}")
    print(f"\nBoxes (float64):\n{boxes_f64}")

    idx = np.array([0, 1], dtype=np.int64)

    # These should intersect (touching boxes are considered intersecting)
    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree intersections: {pairs_f64}")
    print(f"Float32 tree intersections: {pairs_f32}")

    # Check consistency
    if len(pairs_f64) != len(pairs_f32):
        print(f"\n⚠️  INCONSISTENCY DETECTED!")
        print(f"Float64 found {len(pairs_f64)} pairs, but float32 found {len(pairs_f32)} pairs")
        return True

    return False


if __name__ == "__main__":
    print("Testing float32 rounding false negative issue in overlap detection")
    print("=" * 70)

    issue_found = False

    issue_found |= test_float32_false_negative()
    issue_found |= test_float32_precision_boundary()
    issue_found |= test_float32_query_method()
    issue_found |= test_float32_ulp_case()

    print("\n" + "=" * 70)
    if issue_found:
        print("❌ ISSUE CONFIRMED: Float32 rounding causes false negatives in overlap detection")
    else:
        print("✅ No false negatives detected in these test cases")
    print("=" * 70)
