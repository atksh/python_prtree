"""Unit tests for PRTree properties and utility methods."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestSizeProperty:
    """Test size() method."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_size_empty_tree(self, PRTree, dim):
        """Verify that size of empty tree is 0Verify that."""
        tree = PRTree()
        assert tree.size() == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_size_after_construction(self, PRTree, dim):
        """Verify that size after construction is correct."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_size_after_insert(self, PRTree, dim):
        """Verify that size after insert is correct."""
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
        """Verify that size after erase is correct."""
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
        """Verify that len of empty tree is 0Verify that."""
        tree = PRTree()
        assert len(tree) == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_len_after_construction(self, PRTree, dim):
        """Verify that len after construction is correct."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert len(tree) == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_len_equals_size(self, PRTree, dim):
        """Verify that len and sizematches."""
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
        """Verify that n property of empty tree is 0Verify that."""
        tree = PRTree()
        assert tree.n == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_n_after_construction(self, PRTree, dim):
        """Verify that n property after construction is correct."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.n == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_n_equals_size_and_len(self, PRTree, dim):
        """Verify that n, size, and len all match."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.n == tree.size() == len(tree)
