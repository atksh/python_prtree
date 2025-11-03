"""Unit tests for PRTree erase operations."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestNormalErase:
    """Test normal erase scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_single_element(self, PRTree, dim):
        """1要素の削除が機能することを確認."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, 2 * dim))
        for i in range(2):
            for d in range(dim):
                boxes[i, d] = i
                boxes[i, d + dim] = i + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == 2

        tree.erase(1)
        assert tree.size() == 1

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_multiple_elements(self, PRTree, dim):
        """複数要素の削除が機能することを確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

        # Erase half
        for i in range(n // 2):
            tree.erase(i)

        assert tree.size() == n - n // 2


class TestErrorErase:
    """Test erase with invalid inputs."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_from_empty_tree(self, PRTree, dim):
        """空のツリーからの削除がエラーになることを確認."""
        tree = PRTree()

        with pytest.raises(ValueError):
            tree.erase(1)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_non_existent_index(self, PRTree, dim):
        """存在しないインデックスの削除の動作確認."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, 2 * dim))
        for i in range(2):
            for d in range(dim):
                boxes[i, d] = i
                boxes[i, d + dim] = i + 1

        tree = PRTree(idx, boxes)

        # Try to erase non-existent index
        # Implementation may raise error or silently fail
        try:
            tree.erase(999)
            # If no error, size should remain same
            assert tree.size() == 2
        except (ValueError, RuntimeError, KeyError):
            # Error is also acceptable
            pass


class TestConsistencyErase:
    """Test erase consistency."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_after_erase(self, PRTree, dim):
        """削除後のクエリが正しい結果を返すことを確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Erase element 0
        tree.erase(0)

        # Query should not return erased element
        query_box = boxes[0]
        result = tree.query(query_box)
        assert 0 not in result

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_after_erase(self, PRTree, dim):
        """削除後の挿入が機能することを確認."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, 2 * dim))
        for i in range(2):
            for d in range(dim):
                boxes[i, d] = i
                boxes[i, d + dim] = i + 1

        tree = PRTree(idx, boxes)

        # Erase then insert
        tree.erase(1)
        assert tree.size() == 1

        new_box = np.zeros(2 * dim)
        for d in range(dim):
            new_box[d] = 10.0
            new_box[d + dim] = 11.0

        tree.insert(idx=3, bb=new_box)
        assert tree.size() == 2
