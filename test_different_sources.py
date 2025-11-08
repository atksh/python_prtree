#!/usr/bin/env python3
"""
Test for rounding issues when values come from different sources.

The hypothesis: When Box A's max and Box B's min are computed or stored
independently, float32 rounding can create gaps that don't exist in float64.
"""
import numpy as np
from python_prtree import PRTree2D


def test_computed_vs_literal():
    """
    Test when coordinates come from computations vs literals.

    A computed value might round differently than a literal value
    due to intermediate precision.
    """
    print("\n=== Test: Computed vs Literal Values ===\n")

    # Computed value (with intermediate float64 precision)
    computed_f64 = np.float64(1.0) / np.float64(3.0) * np.float64(300.0)  # = 100.0

    # Literal value
    literal_f64 = np.float64(100.0)

    print(f"Computed (f64): {computed_f64:.20f}")
    print(f"Literal (f64):  {literal_f64:.20f}")
    print(f"Equal in f64: {computed_f64 == literal_f64}")

    computed_f32 = np.float32(computed_f64)
    literal_f32 = np.float32(literal_f64)

    print(f"\nComputed (f32): {computed_f32:.20f}")
    print(f"Literal (f32):  {literal_f32:.20f}")
    print(f"Equal in f32: {computed_f32 == literal_f32}")

    # Create boxes
    boxes = np.array([
        [0.0, 0.0, computed_f64, 100.0],  # Ends at computed value
        [literal_f64, 0.0, 200.0, 100.0],  # Starts at literal value
    ], dtype=np.float64)

    boxes_f32 = boxes.astype(np.float32)

    print(f"\nOverlap (f64): {boxes[0,2] >= boxes[1,0]}")
    print(f"Overlap (f32): {boxes_f32[0,2] >= boxes_f32[1,0]}")

    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32 = PRTree2D(idx, boxes_f32)
    pairs_f32 = tree_f32.query_intersections()

    print(f"\nFloat64 tree: {len(pairs_f64)} pairs")
    print(f"Float32 tree: {len(pairs_f32)} pairs")

    if len(pairs_f64) != len(pairs_f32):
        print("\n❌ FALSE NEGATIVE!")
        return True
    return False


def test_accumulated_computation():
    """
    Test when values are accumulated through multiple operations.
    """
    print("\n=== Test: Accumulated Computation ===\n")

    # Create a value through accumulation
    accumulated_f64 = np.float64(0.0)
    step = np.float64(0.1)
    for i in range(1000):
        accumulated_f64 += step

    # Direct value
    direct_f64 = np.float64(100.0)

    print(f"Accumulated (f64): {accumulated_f64:.20f}")
    print(f"Direct (f64):      {direct_f64:.20f}")
    print(f"Difference:        {abs(accumulated_f64 - direct_f64):.20e}")

    accumulated_f32 = np.float32(accumulated_f64)
    direct_f32 = np.float32(direct_f64)

    print(f"\nAccumulated (f32): {accumulated_f32:.20f}")
    print(f"Direct (f32):      {direct_f32:.20f}")
    print(f"Equal in f32: {accumulated_f32 == direct_f32}")

    # Create boxes with these values
    boxes_f64 = np.array([
        [0.0, 0.0, accumulated_f64, 100.0],
        [direct_f64, 0.0, 200.0, 100.0],
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

    if len(pairs_f64) != len(pairs_f32):
        print("\n❌ FALSE NEGATIVE!")
        return True
    return False


def test_separate_float32_arrays():
    """
    Test when float32 values are created in separate arrays.

    Key insight: If two float32 values are created independently,
    they might have different representations even if they should be equal.
    """
    print("\n=== Test: Separate Float32 Arrays ===\n")

    # Create a problematic float64 value
    problematic = np.float64(100.0) + np.float64(1e-7)

    print(f"Problematic value (f64): {problematic:.20f}")

    # Create first array with this value as max
    array1_f32 = np.array([0.0, 0.0, problematic, 100.0], dtype=np.float32)

    # Create second array with this value as min
    array2_f32 = np.array([problematic, 0.0, 200.0, 100.0], dtype=np.float32)

    print(f"\nArray1[2] (max): {array1_f32[2]:.20f}")
    print(f"Array2[0] (min): {array2_f32[0]:.20f}")
    print(f"Equal: {array1_f32[2] == array2_f32[0]}")
    print(f"Overlap: {array1_f32[2] >= array2_f32[0]}")

    # Combine into boxes
    boxes_f32 = np.vstack([array1_f32.reshape(1, -1), array2_f32.reshape(1, -1)])

    print(f"\nCombined boxes (f32):\n{boxes_f32}")

    idx = np.array([0, 1], dtype=np.int64)

    tree = PRTree2D(idx, boxes_f32)
    pairs = tree.query_intersections()

    print(f"\nIntersections found: {len(pairs)}")
    print(f"Pairs: {pairs}")

    # Expected: should find intersection since they touch
    if len(pairs) == 0:
        print("\n❌ FALSE NEGATIVE: Touching boxes not detected!")
        return True

    return False


def test_binary_representation():
    """
    Test values that have identical decimal representation but different binary.
    """
    print("\n=== Test: Binary Representation ===\n")

    # Create a value that cannot be exactly represented in binary
    decimal_val = 0.1

    # In float64
    val_f64 = np.float64(decimal_val)
    print(f"0.1 in float64: {val_f64:.60f}")
    print(f"Hex: {val_f64.hex()}")

    # In float32
    val_f32 = np.float32(decimal_val)
    print(f"\n0.1 in float32: {val_f32:.60f}")
    print(f"Hex: {val_f32.hex()}")

    # Scale up
    scale = 1000
    scaled_f64 = val_f64 * scale
    scaled_f32 = val_f32 * scale

    print(f"\nScaled (f64): {scaled_f64:.60f}")
    print(f"Scaled (f32): {scaled_f32:.60f}")

    # Now use these as coordinates
    boxes_f64 = np.array([
        [0.0, 0.0, scaled_f64, 100.0],
        [scaled_f64, 0.0, 200.0, 100.0],
    ], dtype=np.float64)

    # Create float32 version two ways:
    # 1. Convert from float64
    boxes_f32_converted = boxes_f64.astype(np.float32)

    # 2. Create directly with float32
    boxes_f32_direct = np.array([
        [0.0, 0.0, val_f32 * scale, 100.0],
        [val_f32 * scale, 0.0, 200.0, 100.0],
    ], dtype=np.float32)

    print(f"\nBoxes f32 (converted):\n{boxes_f32_converted}")
    print(f"\nBoxes f32 (direct):\n{boxes_f32_direct}")
    print(f"\nAre they equal? {np.array_equal(boxes_f32_converted, boxes_f32_direct)}")

    idx = np.array([0, 1], dtype=np.int64)

    tree_f64 = PRTree2D(idx, boxes_f64)
    pairs_f64 = tree_f64.query_intersections()

    tree_f32_conv = PRTree2D(idx, boxes_f32_converted)
    pairs_f32_conv = tree_f32_conv.query_intersections()

    tree_f32_dir = PRTree2D(idx, boxes_f32_direct)
    pairs_f32_dir = tree_f32_dir.query_intersections()

    print(f"\nFloat64 tree: {len(pairs_f64)} pairs")
    print(f"Float32 (converted): {len(pairs_f32_conv)} pairs")
    print(f"Float32 (direct): {len(pairs_f32_dir)} pairs")

    if len(pairs_f64) != len(pairs_f32_conv) or len(pairs_f64) != len(pairs_f32_dir):
        print("\n❌ FALSE NEGATIVE!")
        return True

    return False


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Different Sources of Rounding")
    print("=" * 70)

    issue_found = False

    issue_found |= test_computed_vs_literal()
    issue_found |= test_accumulated_computation()
    issue_found |= test_separate_float32_arrays()
    issue_found |= test_binary_representation()

    print("\n" + "=" * 70)
    if issue_found:
        print("❌ FALSE NEGATIVE CONFIRMED!")
    else:
        print("⚠️  No false negatives in these tests")
    print("=" * 70)
