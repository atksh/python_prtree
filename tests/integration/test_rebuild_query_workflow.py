"""Integration tests for rebuild → query workflow."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_rebuild_after_many_operations(PRTree, dim):
    """Test rebuild and query after many operations."""
    np.random.seed(42)
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree = PRTree(idx, boxes)

    # Many insert operations
    for i in range(n, n + 100):
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1
        tree.insert(idx=i, bb=box)

    # Many erase operations
    for i in range(n // 2):
        tree.erase(i)

    # Rebuild
    tree.rebuild()

    # Query should still work
    query_box = np.random.rand(2 * dim) * 100
    for i in range(dim):
        query_box[i + dim] += query_box[i] + 1

    result = tree.query(query_box)
    assert isinstance(result, list)


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_rebuild_consistency_across_operations(PRTree, dim):
    """Test consistency before and after rebuild."""
    np.random.seed(42)
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree1 = PRTree(idx, boxes)
    tree2 = PRTree(idx, boxes)

    # Tree1: many operations + rebuild
    for i in range(20):
        tree1.erase(i)
    for i in range(20):
        box = boxes[i]
        tree1.insert(idx=i, bb=box)
    tree1.rebuild()

    # Query both trees
    queries = np.random.rand(20, 2 * dim) * 100
    for i in range(dim):
        queries[:, i + dim] += queries[:, i] + 1

    results1 = tree1.batch_query(queries)
    results2 = tree2.batch_query(queries)

    # Results should be identical
    for r1, r2 in zip(results1, results2):
        assert set(r1) == set(r2)
