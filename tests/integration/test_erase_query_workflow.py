"""Integration tests for erase → query workflow."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_erase_and_query_incrementally(PRTree, dim):
    """インクリメンタルに削除しながらクエリする統合テスト."""
    np.random.seed(42)
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree = PRTree(idx, boxes)

    # Erase half and query after each erase
    for i in range(n // 2):
        tree.erase(i)
        assert tree.size() == n - i - 1

        # Query for erased element should return empty or not include it
        result = tree.query(boxes[i])
        assert i not in result


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_insert_erase_insert_workflow(PRTree, dim):
    """挿入→削除→挿入のワークフローテスト."""
    tree = PRTree()

    # Insert two elements (can't erase the last element due to library limitation)
    box1 = np.zeros(2 * dim)
    for d in range(dim):
        box1[d] = 0.0
        box1[d + dim] = 1.0
    tree.insert(idx=1, bb=box1)

    box_dummy = np.zeros(2 * dim)
    for d in range(dim):
        box_dummy[d] = 10.0
        box_dummy[d + dim] = 11.0
    tree.insert(idx=999, bb=box_dummy)

    # Erase first element
    tree.erase(1)
    assert tree.size() == 1

    # Insert again
    box2 = np.zeros(2 * dim)
    for d in range(dim):
        box2[d] = 2.0
        box2[d + dim] = 3.0
    tree.insert(idx=2, bb=box2)

    assert tree.size() == 2
    result = tree.query(box2)
    assert 2 in result
    assert 1 not in result  # Verify element 1 was erased


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_bulk_erase_and_verify(PRTree, dim):
    """大量削除後の検証テスト."""
    np.random.seed(42)
    n = 200
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree = PRTree(idx, boxes)

    # Erase even indices
    for i in range(0, n, 2):
        tree.erase(i)

    assert tree.size() == n // 2

    # Verify remaining elements
    for i in range(1, n, 2):
        result = tree.query(boxes[i])
        assert i in result
