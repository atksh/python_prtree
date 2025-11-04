"""Unit tests for PRTree insert operations."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestNormalInsert:
    """Test normal insert scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_single_element(self, PRTree, dim):
        """Verify that single element insertworks."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        tree.insert(idx=1, bb=box)
        assert tree.size() == 1

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_multiple_elements(self, PRTree, dim):
        """Verify that multiple element insertworks."""
        tree = PRTree()

        for i in range(10):
            box = np.zeros(2 * dim)
            for d in range(dim):
                box[d] = i
                box[d + dim] = i + 1

            tree.insert(idx=i, bb=box)

        assert tree.size() == 10

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_auto_index(self, PRTree, dim):
        """Verify that insert with auto indexworks."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        # Insert without specifying idx (should auto-generate)
        tree.insert(bb=box, obj={"data": "test"})
        assert tree.size() == 1

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_object(self, PRTree, dim):
        """Verify that insert with objectworks."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        obj = {"name": "test", "value": 123}
        tree.insert(idx=1, bb=box, obj=obj)

        assert tree.size() == 1

        # Query and retrieve object
        result = tree.query(box, return_obj=True)
        assert len(result) == 1
        assert result[0] == obj


class TestErrorInsert:
    """Test insert with invalid inputs."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_without_box(self, PRTree, dim):
        """Verify that insert without boxraises an error."""
        tree = PRTree()

        with pytest.raises(ValueError):
            tree.insert(idx=1)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_without_index_and_object(self, PRTree, dim):
        """Verify that insert without index and objectraises an error."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        with pytest.raises(ValueError):
            tree.insert(bb=box)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_invalid_box(self, PRTree, dim):
        """Verify that insert with invalid box (min > max)raises an error."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 10.0  # min
            box[i + dim] = 0.0  # max (invalid)

        with pytest.raises((ValueError, RuntimeError)):
            tree.insert(idx=1, bb=box)


class TestConsistencyInsert:
    """Test insert consistency."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_after_insert(self, PRTree, dim):
        """Verify that query after insert returns correct results."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Insert new element
        new_box = np.zeros(2 * dim)
        for i in range(dim):
            new_box[i] = 50.0
            new_box[i + dim] = 60.0

        tree.insert(idx=n, bb=new_box)
        assert tree.size() == n + 1

        # Query for new element
        result = tree.query(new_box)
        assert n in result

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_incremental_construction(self, PRTree, dim):
        """Verify that incremental build returns same results as bulk build."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        # Bulk construction
        tree1 = PRTree(idx, boxes)

        # Incremental construction
        tree2 = PRTree()
        for i in range(n):
            tree2.insert(idx=idx[i], bb=boxes[i])

        # Query both trees
        query_box = np.random.rand(2 * dim) * 100
        for i in range(dim):
            query_box[i + dim] += query_box[i] + 1

        result1 = tree1.query(query_box)
        result2 = tree2.query(query_box)

        assert set(result1) == set(result2)
