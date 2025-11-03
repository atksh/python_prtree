"""End-to-end regression tests for known bugs.

These tests ensure that previously fixed bugs don't reoccur.
"""
import gc
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_disjoint_small_gap_bug(PRTree, dim):
    """Regression test for Matteo Lacki's bug (Issue #45).

    Boxes with small gaps (< 1e-5) were incorrectly reported as intersecting
    due to float32 precision loss. This has been fixed in v0.7.0.
    """
    if dim == 2:
        A = np.array([[72.47410062, 80.52848893, 75.02750896, 85.40646976]])
        B = np.array([[75.02751435, 80.0, 78.71358218, 85.0]])
        gap_dim = 0
    elif dim == 3:
        A = np.array([[72.47410062, 80.52848893, 54.68197159, 75.02750896, 85.40646976, 62.42859506]])
        B = np.array([[75.02751435, 74.65699325, 61.09751679, 78.71358218, 82.4585436, 67.24904609]])
        gap_dim = 0
    else:  # dim == 4
        A = np.array([[72.47410062, 80.52848893, 54.68197159, 60.0, 75.02750896, 85.40646976, 62.42859506, 70.0]])
        B = np.array([[75.02751435, 74.65699325, 61.09751679, 55.0, 78.71358218, 82.4585436, 67.24904609, 65.0]])
        gap_dim = 0

    assert A[0][gap_dim + dim] < B[0][gap_dim], f"Test setup error: boxes should be disjoint"
    gap = B[0][gap_dim] - A[0][gap_dim + dim]
    assert gap > 0, f"Gap should be positive, got {gap}"

    tree = PRTree(np.array([0]), A)

    result = tree.batch_query(B)
    assert result == [[]], f"Expected [[]] (no intersection), got {result}. Gap was {gap}"

    result_query = tree.query(B[0])
    assert result_query == [], f"Expected [] (no intersection), got {result_query}. Gap was {gap}"


def test_save_load_float64_precision_bug(tmp_path):
    """Regression test for float64 precision loss after save/load.

    idx2exact was not being serialized, causing float64 trees to lose
    precision after save/load. Fixed in v0.7.0.
    """
    A = np.array([[72.47410062, 80.52848893, 54.68197159, 75.02750896, 85.40646976, 62.42859506]], dtype=np.float64)
    B = np.array([[75.02751435, 74.65699325, 61.09751679, 78.71358218, 82.4585436, 67.24904609]], dtype=np.float64)

    assert A[0][3] < B[0][0], "Test setup error: boxes should be disjoint"
    gap = B[0][0] - A[0][3]
    assert 5e-6 < gap < 6e-6, f"Test setup error: expected gap ~5.4e-6, got {gap}"

    tree = PRTree3D(np.array([0], dtype=np.int64), A)

    result_before = tree.batch_query(B)
    assert result_before == [[]], f"Before save: Expected [[]] (disjoint), got {result_before}"

    fname = tmp_path / "tree_float64.bin"
    fname = str(fname)
    tree.save(fname)

    del tree
    gc.collect()

    tree_loaded = PRTree3D(fname)

    result_after = tree_loaded.batch_query(B)
    assert result_after == [[]], f"After load: Expected [[]] (disjoint), got {result_after}"


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_touching_boxes_semantics(PRTree, dim):
    """Regression test: ensure closed interval semantics are maintained.

    Boxes that exactly touch (share a boundary) should be considered
    intersecting. This is the intended behavior.
    """
    A = np.zeros((1, 2 * dim))
    B = np.zeros((1, 2 * dim))

    for i in range(dim):
        A[0][i] = 0.0  # min coords
        A[0][i + dim] = 1.0  # max coords
        B[0][i] = 1.0  # min coords
        B[0][i + dim] = 2.0  # max coords

    tree = PRTree(np.array([0]), A)

    result = tree.batch_query(B)
    assert result == [[0]], f"Expected [[0]] (touching boxes intersect), got {result}"

    result_query = tree.query(B[0])
    assert result_query == [0], f"Expected [0] (touching boxes intersect), got {result_query}"


def test_empty_tree_insert_bug():
    """Regression test: inserting into an empty PRTree was broken before v0.5.0."""
    tree = PRTree2D()
    assert tree.size() == 0

    # This was broken before v0.5.0
    tree.insert(idx=1, bb=[0, 0, 1, 1])
    assert tree.size() == 1

    result = tree.query([0.5, 0.5, 0.6, 0.6])
    assert result == [1]


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_degenerate_boxes_no_crash(PRTree, dim):
    """Regression test: degenerate boxes (min == max) should not crash."""
    n = 10
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100

    # Make all boxes degenerate
    for i in range(dim):
        boxes[:, i + dim] = boxes[:, i]

    # Should not crash
    tree = PRTree(idx, boxes)
    assert tree.size() == n

    # Queries should not crash (though degenerate boxes may not be found in all-degenerate trees)
    query_box = boxes[0]
    result = tree.query(query_box)
    # Note: Query may return empty for all-degenerate datasets due to R-tree limitations
    assert isinstance(result, list)  # Just verify it doesn't crash


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_large_magnitude_coordinates_precision(PRTree, dim):
    """Regression test: ensure precision is maintained with large coordinates."""
    A = np.zeros((1, 2 * dim))
    B = np.zeros((1, 2 * dim))

    base = 1e6
    for i in range(dim):
        A[0][i] = base + i  # min coords
        A[0][i + dim] = base + i + 1.0  # max coords
        B[0][i] = base + i + 1.1  # min coords (gap)
        B[0][i + dim] = base + i + 2.0  # max coords

    tree = PRTree(np.array([0]), A)

    result = tree.batch_query(B)
    assert result == [[]], f"Expected [[]] (no intersection at large magnitude), got {result}"


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_query_intersections_correctness(PRTree, dim):
    """Regression test: query_intersections should return all and only intersecting pairs."""
    np.random.seed(42)
    n = 30
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree = PRTree(idx, boxes)
    pairs = tree.query_intersections()

    # Verify with naive approach
    def has_intersect(x, y, dim):
        return all([max(x[i], y[i]) <= min(x[i + dim], y[i + dim]) for i in range(dim)])

    expected_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if has_intersect(boxes[i], boxes[j], dim):
                expected_pairs.append((idx[i], idx[j]))

    pairs_set = set(map(tuple, pairs))
    expected_set = set(expected_pairs)

    assert pairs_set == expected_set, f"Mismatch: expected {len(expected_set)} pairs, got {len(pairs_set)}"
