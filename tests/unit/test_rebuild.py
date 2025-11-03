"""Unit tests for PRTree rebuild operations."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestNormalRebuild:
    """Test normal rebuild scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_rebuild_after_construction(self, PRTree, dim):
        """構築後のrebuildが機能することを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        tree.rebuild()

        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_rebuild_after_insert(self, PRTree, dim):
        """挿入後のrebuildが機能することを確認."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Insert more elements
        for i in range(n, n + 50):
            box = np.random.rand(2 * dim) * 100
            for d in range(dim):
                box[d + dim] += box[d] + 1
            tree.insert(idx=i, bb=box)

        tree.rebuild()
        assert tree.size() == n + 50

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_rebuild_after_erase(self, PRTree, dim):
        """削除後のrebuildが機能することを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Erase half
        for i in range(n // 2):
            tree.erase(i)

        tree.rebuild()
        assert tree.size() == n - n // 2


class TestConsistencyRebuild:
    """Test rebuild consistency."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_results_before_after_rebuild(self, PRTree, dim):
        """rebuild前後でクエリ結果が一致することを確認."""
        np.random.seed(42)
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Query before rebuild
        query_box = np.random.rand(2 * dim) * 100
        for i in range(dim):
            query_box[i + dim] += query_box[i] + 1

        result_before = tree.query(query_box)

        # Rebuild
        tree.rebuild()

        # Query after rebuild
        result_after = tree.query(query_box)

        assert set(result_before) == set(result_after)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_multiple_rebuilds(self, PRTree, dim):
        """複数回のrebuildが機能することを確認."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        query_box = np.random.rand(2 * dim) * 100
        for i in range(dim):
            query_box[i + dim] += query_box[i] + 1

        result_initial = tree.query(query_box)

        # Multiple rebuilds
        for _ in range(3):
            tree.rebuild()
            result = tree.query(query_box)
            assert set(result) == set(result_initial)
