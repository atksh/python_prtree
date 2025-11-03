"""Unit tests for PRTree erase operations."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestNormalErase:
    """Test normal erase scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_single_element(self, PRTree, dim):
        """Verify that single element eraseworks."""
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
        """Verify that multiple element eraseworks."""
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
        """Verify that erase from empty treeraises an error."""
        tree = PRTree()

        with pytest.raises(ValueError):
            tree.erase(1)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_non_existent_index(self, PRTree, dim):
        """Verify that erase of non-existent indexraises an error."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, 2 * dim))
        for i in range(2):
            for d in range(dim):
                boxes[i, d] = i
                boxes[i, d + dim] = i + 1

        tree = PRTree(idx, boxes)

        # Try to erase non-existent index - should raise error
        with pytest.raises(RuntimeError, match="Given index is not found"):
            tree.erase(999)

        # Tree should be unchanged
        assert tree.size() == 2

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_non_existent_index_single_element(self, PRTree, dim):
        """Verify that erase of non-existent index in single-element tree raises an error (P1 validation bug)."""
        idx = np.array([5])
        boxes = np.zeros((1, 2 * dim))
        for d in range(dim):
            boxes[0, d] = 0.0
            boxes[0, d + dim] = 1.0

        tree = PRTree(idx, boxes)
        assert tree.size() == 1

        # Try to erase non-existent index 999 - should raise error
        # This is the P1 bug: previously silently deleted the real element
        with pytest.raises(RuntimeError, match="Given index is not found"):
            tree.erase(999)

        # Tree should still contain the element
        assert tree.size() == 1

        # Verify the correct element is still there
        query_box = boxes[0]
        result = tree.query(query_box)
        assert 5 in result

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_valid_index_single_element(self, PRTree, dim):
        """Verify that erase of valid index in single-element treeworks."""
        idx = np.array([5])
        boxes = np.zeros((1, 2 * dim))
        for d in range(dim):
            boxes[0, d] = 0.0
            boxes[0, d + dim] = 1.0

        tree = PRTree(idx, boxes)
        assert tree.size() == 1

        # Erase the valid index 5 - should succeed
        tree.erase(5)

        # Tree should now be empty
        assert tree.size() == 0


class TestConsistencyErase:
    """Test erase consistency."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_after_erase(self, PRTree, dim):
        """Verify that query after erase returns correct results."""
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
        """Verify that insert after eraseworks."""
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
