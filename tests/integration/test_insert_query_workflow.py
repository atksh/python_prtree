"""Integration tests for insert → query workflow."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_incremental_insert_and_query(PRTree, dim):
    """Integration test: incremental insert with queries."""
    tree = PRTree()

    n = 100
    boxes = []

    for i in range(n):
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1

        tree.insert(idx=i, bb=box)
        boxes.append(box)

        # Query after each insert
        result = tree.query(box)
        assert i in result
        assert tree.size() == i + 1

    # Final comprehensive query
    for i, box in enumerate(boxes):
        result = tree.query(box)
        assert i in result


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_insert_with_objects_and_query(PRTree, dim):
    """Integration test: insert with objects and query."""
    tree = PRTree()

    n = 50
    objects = []

    for i in range(n):
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1

        obj = {"id": i, "name": f"item_{i}", "data": [i, i * 2, i * 3]}
        tree.insert(bb=box, obj=obj)
        objects.append((box, obj))

    # Query and verify objects (return_obj=True returns objects directly, not tuples)
    for i, (box, expected_obj) in enumerate(objects):
        result_obj = tree.query(box, return_obj=True)
        found = False
        for item in result_obj:
            if item == expected_obj:
                found = True
                break
        # Object retrieval should return the inserted object
        assert len(result_obj) > 0
        assert found, f"Expected object {expected_obj} not found in results"


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_mixed_bulk_and_incremental_insert(PRTree, dim):
    """Test mixed bulk and incremental insert."""
    np.random.seed(42)
    n_bulk = 50
    n_incremental = 50

    # Bulk insert
    idx_bulk = np.arange(n_bulk)
    boxes_bulk = np.random.rand(n_bulk, 2 * dim) * 100
    for i in range(dim):
        boxes_bulk[:, i + dim] += boxes_bulk[:, i] + 1

    tree = PRTree(idx_bulk, boxes_bulk)

    # Incremental insert
    for i in range(n_incremental):
        idx = n_bulk + i
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1

        tree.insert(idx=idx, bb=box)

    assert tree.size() == n_bulk + n_incremental

    # Query all
    query_box = np.zeros(2 * dim)
    for d in range(dim):
        query_box[d] = -10.0
        query_box[d + dim] = 110.0

    result = tree.query(query_box)
    assert len(result) == n_bulk + n_incremental
