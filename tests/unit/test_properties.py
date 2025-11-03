"""Unit tests for PRTree properties and utility methods."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestSizeProperty:
    """Test size() method."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_size_empty_tree(self, PRTree, dim):
        """空のツリーのサイズが0であることを確認."""
        tree = PRTree()
        assert tree.size() == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_size_after_construction(self, PRTree, dim):
        """構築後のサイズが正しいことを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_size_after_insert(self, PRTree, dim):
        """挿入後のサイズが正しいことを確認."""
        tree = PRTree()

        for i in range(10):
            box = np.zeros(2 * dim)
            for d in range(dim):
                box[d] = i
                box[d + dim] = i + 1
            tree.insert(idx=i, bb=box)
            assert tree.size() == i + 1

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_size_after_erase(self, PRTree, dim):
        """削除後のサイズが正しいことを確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        for i in range(n):
            tree.erase(i)
            assert tree.size() == n - i - 1


class TestLenProperty:
    """Test __len__() method."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_len_empty_tree(self, PRTree, dim):
        """空のツリーのlenが0であることを確認."""
        tree = PRTree()
        assert len(tree) == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_len_after_construction(self, PRTree, dim):
        """構築後のlenが正しいことを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert len(tree) == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_len_equals_size(self, PRTree, dim):
        """lenとsizeが一致することを確認."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert len(tree) == tree.size()


class TestNProperty:
    """Test n property."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_n_empty_tree(self, PRTree, dim):
        """空のツリーのnプロパティが0であることを確認."""
        tree = PRTree()
        assert tree.n == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_n_after_construction(self, PRTree, dim):
        """構築後のnプロパティが正しいことを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.n == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_n_equals_size_and_len(self, PRTree, dim):
        """n、size、lenが全て一致することを確認."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.n == tree.size() == len(tree)
