#!/usr/bin/env python3
"""
Test to find actual false negatives in float32 overlap detection.

Based on the insight from accumulated computation test, we need to find
cases where:
- Float64: overlap exists
- Float32: overlap lost due to rounding
"""
import numpy as np
from python_prtree import PRTree2D


def test_accumulated_overlap_loss():
    """
    Test where accumulated computation creates a value that overlaps in float64
    but loses overlap when rounded to float32.
    """
    print("\n=== Test: Accumulated Overlap Loss ===\n")

    # Create a value through accumulation that is slightly larger than direct
    accumulated_f64 = np.float64(0.0)
    step = np.float64(0.1)
    for i in range(1001):  # 1001 * 0.1 = 100.1 (approximately)
        accumulated_f64 += step

    direct_f64 = np.float64(100.0)

    print(f"Accumulated (f64): {accumulated_f64:.20f}")
    print(f"Direct (f64):      {direct_f64:.20f}")
    print(f"accumulated > direct: {accumulated_f64 > direct_f64}")
    print(f"Difference: {accumulated_f64 - direct_f64:.20e}")

    # Convert to float32
    accumulated_f32 = np.float32(accumulated_f64)
    direct_f32 = np.float32(direct_f64)

    print(f"\nAccumulated (f32): {accumulated_f32:.20f}")
    print(f"Direct (f32):      {direct_f32:.20f}")
    print(f"accumulated > direct: {accumulated_f32 > direct_f32}")

    # Create boxes:
    # Box A: ends at accumulated value (slightly > 100 in f64)
    # Box B: starts at 100.0
    # In f64: overlap (accumulated >= 100)
    # In f32: might not overlap if both round to same value but lose the "greater than" relationship

    boxes_f64 = np.array([
        [0.0, 0.0, accumulated_f64, 100.0],  # Box A
        [direct_f64, 0.0, 200.0, 100.0],      # Box B
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nBoxes (f64):")
    print(f"  Box A ends at:   {boxes_f64[0,2]:.20f}")
    print(f"  Box B starts at: {boxes_f64[1,0]:.20f}")
    print(f"  Overlap: {boxes_f64[0,2] >= boxes_f64[1,0]}")

    print(f"\nBoxes (f32):")
    print(f"  Box A ends at:   {boxes_f32[0,2]:.20f}")
    print(f"  Box B starts at: {boxes_f32[1,0]:.20f}")
    print(f"  Overlap: {boxes_f32[0,2] >= boxes_f32[1,0]}")

    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree: {len(pairs_f64)} pairs - {pairs_f64}")
    print(f"Float32 tree: {len(pairs_f32)} pairs - {pairs_f32}")

    if len(pairs_f64) > 0 and len(pairs_f32) == 0:
        print("\n❌ FALSE NEGATIVE DETECTED!")
        print("   Float64 found overlap, but Float32 did not!")
        return True

    return False


def test_different_accumulation_rates():
    """
    Test with different accumulation rates to find rounding mismatches.
    """
    print("\n=== Test: Different Accumulation Rates ===\n")

    # Accumulate from below
    from_below_f64 = np.float64(0.0)
    for i in range(1000):
        from_below_f64 += np.float64(0.1000001)  # Slightly more than 0.1

    # Target value
    target_f64 = np.float64(100.0)

    print(f"From below (f64): {from_below_f64:.20f}")
    print(f"Target (f64):     {target_f64:.20f}")
    print(f"from_below >= target: {from_below_f64 >= target_f64}")

    from_below_f32 = np.float32(from_below_f64)
    target_f32 = np.float32(target_f64)

    print(f"\nFrom below (f32): {from_below_f32:.20f}")
    print(f"Target (f32):     {target_f32:.20f}")
    print(f"from_below >= target: {from_below_f32 >= target_f32}")

    # Create boxes
    boxes_f64 = np.array([
        [0.0, 0.0, from_below_f64, 100.0],
        [target_f64, 0.0, 200.0, 100.0],
    ], dtype=np.float64)

    boxes_f32 = boxes_f64.astype(np.float32)

    print(f"\nOverlap (f64): {boxes_f64[0,2] >= boxes_f64[1,0]}")
    print(f"Overlap (f32): {boxes_f32[0,2] >= boxes_f32[1,0]}")

    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree: {len(pairs_f64)} pairs")
    print(f"Float32 tree: {len(pairs_f32)} pairs")

    if len(pairs_f64) > 0 and len(pairs_f32) == 0:
        print("\n❌ FALSE NEGATIVE DETECTED!")
        return True

    return False


def test_nextafter_gap():
    """
    Use nextafter to create values that are adjacent in float64 but not in float32.
    """
    print("\n=== Test: NextAfter Gap ===\n")

    # Start with a float32 value
    base_f32 = np.float32(1000.0)
    next_f32 = np.nextafter(base_f32, np.float32(np.inf))

    print(f"base_f32: {base_f32}")
    print(f"next_f32: {next_f32}")
    print(f"Gap in f32: {next_f32 - base_f32}")

    # In float64, create many values between them
    base_f64 = np.float64(base_f32)
    next_f64 = np.float64(next_f32)

    # Create a float64 value just slightly above base
    epsilon_f64 = (next_f64 - base_f64) / 1000.0  # Much smaller than f32 gap
    val1_f64 = base_f64 + epsilon_f64

    print(f"\nbase_f64:    {base_f64:.20f}")
    print(f"val1_f64:    {val1_f64:.20f}")
    print(f"next_f64:    {next_f64:.20f}")
    print(f"val1 > base: {val1_f64 > base_f64}")

    # Convert to float32
    val1_f32 = np.float32(val1_f64)

    print(f"\nval1_f32: {val1_f32}")
    print(f"val1_f32 == base_f32: {val1_f32 == base_f32}")

    # If val1_f32 rounds down to base_f32, we have a problem
    if val1_f64 > base_f64 and val1_f32 == base_f32:
        print("\n⚠️  val1 rounds down to base in float32!")

        # Create boxes:
        # Box A: [0, 0, val1_f64, 100]  -> [0, 0, base_f32, 100]
        # Box B: [base_f64, 0, 2000, 100] -> [base_f32, 0, 2000, 100]
        # In f64: overlap (val1_f64 > base_f64)
        # In f32: touch exactly (base_f32 == base_f32), still should overlap with <=

        boxes_f64 = np.array([
            [0.0, 0.0, val1_f64, 100.0],
            [base_f64, 0.0, 2000.0, 100.0],
        ], dtype=np.float64)

        boxes_f32 = boxes_f64.astype(np.float32)

        print(f"\nBoxes (f64):")
        print(f"  Box A max: {boxes_f64[0,2]:.20f}")
        print(f"  Box B min: {boxes_f64[1,0]:.20f}")
        print(f"  Overlap: {boxes_f64[0,2] >= boxes_f64[1,0]}")

        print(f"\nBoxes (f32):")
        print(f"  Box A max: {boxes_f32[0,2]:.20f}")
        print(f"  Box B min: {boxes_f32[1,0]:.20f}")
        print(f"  Overlap: {boxes_f32[0,2] >= boxes_f32[1,0]}")

        idx = np.array([0, 1], dtype=np.int64)

        tree_f64 = PRTree2D(idx, boxes_f64)
        pairs_f64 = tree_f64.query_intersections()

        tree_f32 = PRTree2D(idx, boxes_f32)
        pairs_f32 = tree_f32.query_intersections()

        print(f"\nFloat64 tree: {len(pairs_f64)} pairs")
        print(f"Float32 tree: {len(pairs_f32)} pairs")

        if len(pairs_f64) > 0 and len(pairs_f32) == 0:
            print("\n❌ FALSE NEGATIVE DETECTED!")
            return True

    return False


def test_str_roundtrip():
    """
    Test values that go through string representation (common in file I/O).
    """
    print("\n=== Test: String Roundtrip ===\n")

    # Create a value, convert to string, and back
    original_f64 = np.float64(100.0000000001)

    # Simulate limited precision string (e.g., from CSV file)
    str_val = f"{original_f64:.10f}"  # 10 decimal places
    roundtrip_f64 = np.float64(str_val)

    print(f"Original (f64):   {original_f64:.20f}")
    print(f"String:           {str_val}")
    print(f"Roundtrip (f64):  {roundtrip_f64:.20f}")

    original_f32 = np.float32(original_f64)
    roundtrip_f32 = np.float32(roundtrip_f64)

    print(f"\nOriginal (f32):   {original_f32:.20f}")
    print(f"Roundtrip (f32):  {roundtrip_f32:.20f}")
    print(f"Equal: {original_f32 == roundtrip_f32}")

    # Create boxes
    boxes_original = np.array([
        [0.0, 0.0, original_f64, 100.0],
        [100.0, 0.0, 200.0, 100.0],
    ], dtype=np.float64)

    boxes_roundtrip = np.array([
        [0.0, 0.0, roundtrip_f64, 100.0],
        [100.0, 0.0, 200.0, 100.0],
    ], dtype=np.float64)

    print(f"\nOriginal overlap (f64): {boxes_original[0,2] >= boxes_original[1,0]}")
    print(f"Roundtrip overlap (f64): {boxes_roundtrip[0,2] >= boxes_roundtrip[1,0]}")

    boxes_f32_orig = boxes_original.astype(np.float32)
    boxes_f32_round = boxes_roundtrip.astype(np.float32)

    print(f"\nOriginal overlap (f32): {boxes_f32_orig[0,2] >= boxes_f32_orig[1,0]}")
    print(f"Roundtrip overlap (f32): {boxes_f32_round[0,2] >= boxes_f32_round[1,0]}")

    return False


if __name__ == "__main__":
    print("=" * 70)
    print("Testing for False Negatives in Float32 Overlap Detection")
    print("=" * 70)

    issue_found = False

    issue_found |= test_accumulated_overlap_loss()
    issue_found |= test_different_accumulation_rates()
    issue_found |= test_nextafter_gap()
    issue_found |= test_str_roundtrip()

    print("\n" + "=" * 70)
    if issue_found:
        print("❌ FALSE NEGATIVE CONFIRMED!")
        print("Float32 rounding causes overlap detection to fail!")
    else:
        print("⚠️  No false negatives detected yet")
        print("The issue may require specific data patterns or computation paths")
    print("=" * 70)
