#!/usr/bin/env python3
"""
Test for false negatives caused by different rounding directions.

When two different float64 values round to different float32 values
in opposite directions, overlapping boxes in float64 may not overlap
in float32.
"""
import numpy as np
from python_prtree import PRTree2D


def find_rounding_direction_mismatch():
    """
    Find two float64 values that:
    1. In float64: v1 >= v2 (overlap)
    2. When converted to float32:
       - v1 rounds DOWN to v1_f32
       - v2 rounds UP to v2_f32
       - v1_f32 < v2_f32 (no overlap!)
    """
    print("\n=== Finding Rounding Direction Mismatch ===\n")

    # Start with a base float32 value
    base_f32 = np.float32(1000.0)
    next_f32 = np.nextafter(base_f32, np.float32(np.inf))

    print(f"base_f32: {base_f32:.20f}")
    print(f"next_f32: {next_f32:.20f}")

    # Convert to float64 to get exact representations
    base_f64 = np.float64(base_f32)
    next_f64 = np.float64(next_f32)

    print(f"\nbase_f64: {base_f64:.20f}")
    print(f"next_f64: {next_f64:.20f}")

    # Create two float64 values in between base and next
    # v1: closer to next (will round UP to next_f32)
    # v2: closer to base (will round DOWN to base_f32)
    # But v1 > v2 in float64

    midpoint = (base_f64 + next_f64) / 2.0

    # v2 slightly below midpoint (rounds down to base_f32)
    v2_f64 = midpoint - (next_f64 - base_f64) * 0.1

    # v1 slightly above v2 but still below midpoint (also rounds down to base_f32)
    v1_f64 = v2_f64 + (next_f64 - base_f64) * 0.05

    print(f"\nmidpoint: {midpoint:.20f}")
    print(f"v1_f64:   {v1_f64:.20f}")
    print(f"v2_f64:   {v2_f64:.20f}")
    print(f"v1_f64 >= v2_f64: {v1_f64 >= v2_f64}")

    # Convert to float32
    v1_f32 = np.float32(v1_f64)
    v2_f32 = np.float32(v2_f64)

    print(f"\nv1_f32: {v1_f32:.20f}")
    print(f"v2_f32: {v2_f32:.20f}")
    print(f"v1_f32 >= v2_f32: {v1_f32 >= v2_f32}")

    if v1_f64 >= v2_f64 and v1_f32 < v2_f32:
        print("\n⚠️  CRITICAL: Rounding direction mismatch found!")
        print(f"   Float64: overlap (v1={v1_f64:.20f} >= v2={v2_f64:.20f})")
        print(f"   Float32: no overlap (v1={v1_f32:.20f} < v2={v2_f32:.20f})")
        return v1_f64, v2_f64, v1_f32, v2_f32

    return None


def test_rounding_direction_false_negative():
    """
    Test if rounding direction causes false negatives in overlap detection.
    """
    print("\n=== Test: Rounding Direction False Negative ===\n")

    # Strategy: Find two float32 values with a gap, then insert float64 values
    # in that gap that will round to different float32 values

    base_f32 = np.float32(100.0)
    next_f32 = np.nextafter(base_f32, np.float32(np.inf))

    gap = next_f32 - base_f32
    print(f"Float32 gap at 100.0: {gap}")

    # Create float64 values in the gap
    base_f64 = np.float64(base_f32)
    next_f64 = np.float64(next_f32)

    # v1: just above base_f64, should round to base_f32
    v1_f64 = base_f64 + (next_f64 - base_f64) * 0.4

    # v2: just below next_f64, should round to next_f32
    v2_f64 = base_f64 + (next_f64 - base_f64) * 0.6

    print(f"\nv1_f64 = {v1_f64:.20f}")
    print(f"v2_f64 = {v2_f64:.20f}")
    print(f"v1_f64 < v2_f64: {v1_f64 < v2_f64}")

    v1_f32 = np.float32(v1_f64)
    v2_f32 = np.float32(v2_f64)

    print(f"\nv1_f32 = {v1_f32:.20f}")
    print(f"v2_f32 = {v2_f32:.20f}")
    print(f"v1_f32 < v2_f32: {v1_f32 < v2_f32}")

    if v1_f32 < v2_f32:
        print(f"\n✓ Rounding created a gap in float32!")
        print(f"  Gap in float64: {v2_f64 - v1_f64:.20e}")
        print(f"  Gap in float32: {v2_f32 - v1_f32:.20e}")

    # Now create boxes:
    # Box A: ends at v1_f64 (will be stored as v1_f32)
    # Box B: starts at v2_f64 (will be stored as v2_f32)
    # In float64: boxes overlap (v1_f64 overlaps with range before v2_f64)
    # In float32: boxes don't overlap (v1_f32 < v2_f32)

    # But wait, for them to overlap in float64, we need v1 >= v2
    # Let's reverse: Box A ends at v2_f64, Box B starts at v1_f64

    boxes_f64 = np.array([
        [0.0, 0.0, v2_f64, 100.0],  # Box A ends at v2
        [v1_f64, 0.0, 200.0, 100.0],  # Box B starts at v1
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nBoxes (float64):")
    print(f"  Box A: [0, 0, {boxes_f64[0,2]:.20f}, 100]")
    print(f"  Box B: [{boxes_f64[1,0]:.20f}, 0, 200, 100]")
    print(f"  Overlap: {boxes_f64[0,2] >= boxes_f64[1,0]}")

    print(f"\nBoxes (float32):")
    print(f"  Box A: [0, 0, {boxes_f32[0,2]:.20f}, 100]")
    print(f"  Box B: [{boxes_f32[1,0]:.20f}, 0, 200, 100]")
    print(f"  Overlap: {boxes_f32[0,2] >= boxes_f32[1,0]}")

    # Test with trees
    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree intersections: {pairs_f64}")
    print(f"Float32 tree intersections: {pairs_f32}")

    if len(pairs_f64) != len(pairs_f32):
        print(f"\n❌ FALSE NEGATIVE DETECTED!")
        print(f"   Float64 found {len(pairs_f64)} pairs")
        print(f"   Float32 found {len(pairs_f32)} pairs")
        return True

    return False


def test_opposite_rounding():
    """
    Test with values that are guaranteed to round in opposite directions.
    """
    print("\n=== Test: Guaranteed Opposite Rounding ===\n")

    # Use a large value where float32 precision is limited
    base = 16777216.0  # 2^24, where float32 ULP = 2.0

    # At this magnitude, float32 can only represent even integers
    # Odd integers will round to the nearest even integer

    # Create two consecutive float32-representable values
    v1_f32 = np.float32(base)      # 16777216.0
    v2_f32 = np.nextafter(v1_f32, np.float32(np.inf))  # 16777218.0

    print(f"v1_f32: {v1_f32}")
    print(f"v2_f32: {v2_f32}")
    print(f"Gap: {v2_f32 - v1_f32}")

    # In float64, create values in between
    v1_f64 = np.float64(v1_f32) + 0.5  # 16777216.5 (rounds down to v1_f32)
    v2_f64 = np.float64(v1_f32) + 1.5  # 16777217.5 (rounds up to v2_f32)

    print(f"\nv1_f64: {v1_f64}")
    print(f"v2_f64: {v2_f64}")

    # Verify rounding
    print(f"\nRounding verification:")
    print(f"  float32(v1_f64) = {np.float32(v1_f64)} (expected {v1_f32})")
    print(f"  float32(v2_f64) = {np.float32(v2_f64)} (expected {v2_f32})")

    # Create overlapping boxes in float64
    # Box A ends at v2_f64 + 0.3 = 16777217.8
    # Box B starts at v1_f64 = 16777216.5
    # They overlap by 1.3 in float64

    box_a_end_f64 = v2_f64 + 0.3
    box_b_start_f64 = v1_f64

    boxes_f64 = np.array([
        [0.0, 0.0, box_a_end_f64, 100.0],
        [box_b_start_f64, 0.0, box_a_end_f64 + 1000, 100.0],
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nBoxes (float64):")
    print(f"  Box A: ends at {boxes_f64[0,2]}")
    print(f"  Box B: starts at {boxes_f64[1,0]}")
    print(f"  Overlap: {boxes_f64[0,2] >= boxes_f64[1,0]}")

    print(f"\nBoxes (float32):")
    print(f"  Box A: ends at {boxes_f32[0,2]}")
    print(f"  Box B: starts at {boxes_f32[1,0]}")
    print(f"  Overlap: {boxes_f32[0,2] >= boxes_f32[1,0]}")

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
    print("=" * 70)
    print("Testing Rounding Direction Mismatch")
    print("=" * 70)

    find_rounding_direction_mismatch()

    issue_found = False
    issue_found |= test_rounding_direction_false_negative()
    issue_found |= test_opposite_rounding()

    print("\n" + "=" * 70)
    if issue_found:
        print("❌ FALSE NEGATIVE CONFIRMED: Rounding direction causes the issue!")
    else:
        print("⚠️  Still investigating rounding direction effects...")
    print("=" * 70)
