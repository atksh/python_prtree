"""Integration tests for complex mixed operations."""
import gc
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_complex_workflow(PRTree, dim, tmp_path):
    """Complex workflow: build→insert→erase→rebuild→save→load→query."""
    np.random.seed(42)
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    # Build
    tree = PRTree(idx, boxes)
    assert tree.size() == n

    # Insert
    for i in range(n, n + 50):
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1
        tree.insert(idx=i, bb=box)

    assert tree.size() == n + 50

    # Erase
    for i in range(n // 2):
        tree.erase(i)

    assert tree.size() == n + 50 - n // 2

    # Rebuild
    tree.rebuild()

    # Query
    query_box = np.random.rand(2 * dim) * 100
    for i in range(dim):
        query_box[i + dim] += query_box[i] + 1

    result_before_save = tree.query(query_box)

    # Save
    fname = tmp_path / "complex_tree.bin"
    tree.save(str(fname))
    del tree
    gc.collect()

    # Load
    loaded_tree = PRTree(str(fname))

    # Query again
    result_after_load = loaded_tree.query(query_box)

    assert set(result_before_save) == set(result_after_load)


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_stress_operations(PRTree, dim):
    """Stress test: massive insert, erase, and query operations."""
    tree = PRTree()

    # Insert 1000 elements
    for i in range(1000):
        box = np.random.rand(2 * dim) * 1000
        for d in range(dim):
            box[d + dim] += box[d] + 1
        tree.insert(idx=i, bb=box)

    assert tree.size() == 1000

    # Random queries
    for _ in range(100):
        query_box = np.random.rand(2 * dim) * 1000
        for d in range(dim):
            query_box[d + dim] += query_box[d] + 1
        result = tree.query(query_box)
        assert isinstance(result, list)

    # Erase half
    for i in range(0, 1000, 2):
        tree.erase(i)

    assert tree.size() == 500

    # More queries
    for _ in range(100):
        query_box = np.random.rand(2 * dim) * 1000
        for d in range(dim):
            query_box[d + dim] += query_box[d] + 1
        result = tree.query(query_box)
        assert isinstance(result, list)


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_query_intersections_after_modifications(PRTree, dim):
    """Test query_intersections after modifications."""
    np.random.seed(42)
    n = 50
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree = PRTree(idx, boxes)

    # Initial intersections
    pairs_initial = tree.query_intersections()

    # Modify tree
    for i in range(10):
        tree.erase(i)

    for i in range(10):
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1
        tree.insert(idx=n + i, bb=box)

    # Query intersections again
    pairs_after = tree.query_intersections()

    # Should return valid pairs
    assert pairs_after.ndim == 2
    assert pairs_after.shape[1] == 2
    if pairs_after.shape[0] > 0:
        assert np.all(pairs_after[:, 0] < pairs_after[:, 1])
