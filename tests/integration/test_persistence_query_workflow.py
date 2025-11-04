"""Integration tests for save → load → query workflow."""
import gc
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_save_load_query_workflow(PRTree, dim, tmp_path):
    """Test save → load → query workflow."""
    np.random.seed(42)
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    # Build and query
    tree = PRTree(idx, boxes)
    query_box = np.random.rand(2 * dim) * 100
    for i in range(dim):
        query_box[i + dim] += query_box[i] + 1

    result_before = tree.query(query_box)

    # Save
    fname = tmp_path / "tree.bin"
    tree.save(str(fname))
    del tree
    gc.collect()

    # Load and query
    loaded_tree = PRTree(str(fname))
    result_after = loaded_tree.query(query_box)

    assert set(result_before) == set(result_after)


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_modify_save_load_workflow(PRTree, dim, tmp_path):
    """Test build → modify → save → load workflow."""
    np.random.seed(42)
    n = 50
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree = PRTree(idx, boxes)

    # Modify: insert and erase
    for i in range(10):
        tree.erase(i)

    new_box = np.random.rand(2 * dim) * 100
    for d in range(dim):
        new_box[d + dim] += new_box[d] + 1
    tree.insert(idx=999, bb=new_box)

    # Save
    fname = tmp_path / "modified_tree.bin"
    tree.save(str(fname))

    # Load and verify
    loaded_tree = PRTree(str(fname))
    assert loaded_tree.size() == tree.size()

    result = loaded_tree.query(new_box)
    assert 999 in result


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_multiple_save_load_cycles(PRTree, dim, tmp_path):
    """Test multiple save → load cycles."""
    np.random.seed(42)
    n = 50
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim).astype(np.float64) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1e-5

    tree = PRTree(idx, boxes)

    queries = np.random.rand(10, 2 * dim).astype(np.float64) * 100
    for i in range(dim):
        queries[:, i + dim] += queries[:, i] + 1e-5

    results = [tree.batch_query(queries)]

    # Multiple cycles
    for cycle in range(3):
        fname = tmp_path / f"tree_cycle_{cycle}.bin"
        tree.save(str(fname))
        del tree
        gc.collect()

        tree = PRTree(str(fname))
        results.append(tree.batch_query(queries))

    # All results should be identical
    for i in range(len(results) - 1):
        assert results[i] == results[i + 1]
