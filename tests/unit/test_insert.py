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

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_nan_coordinates_float32(self, PRTree, dim):
        """Verify that insert with NaN coordinates (float32) raises an error."""
        tree = PRTree()

        box = np.zeros(2 * dim, dtype=np.float32)
        box[0] = np.nan

        with pytest.raises((ValueError, RuntimeError)):
            tree.insert(idx=1, bb=box)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_nan_coordinates_float64(self, PRTree, dim):
        """Verify that insert with NaN coordinates (float64) raises an error."""
        tree = PRTree()

        box = np.zeros(2 * dim, dtype=np.float64)
        box[0] = np.nan

        with pytest.raises((ValueError, RuntimeError)):
            tree.insert(idx=1, bb=box)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_inf_coordinates_float32(self, PRTree, dim):
        """Verify that insert with Inf coordinates (float32) raises an error."""
        tree = PRTree()

        box = np.zeros(2 * dim, dtype=np.float32)
        box[0] = np.inf

        with pytest.raises((ValueError, RuntimeError)):
            tree.insert(idx=1, bb=box)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_inf_coordinates_float64(self, PRTree, dim):
        """Verify that insert with Inf coordinates (float64) raises an error."""
        tree = PRTree()

        box = np.zeros(2 * dim, dtype=np.float64)
        box[0] = np.inf

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


class TestPrecisionInsert:
    """Test insert with precision requirements."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_float64_maintains_precision(self, PRTree, dim):
        """Verify that float64 insert maintains double-precision refinement."""
        # Create tree with float64 construction
        A = np.zeros((1, 2 * dim), dtype=np.float64)
        A[0, 0] = 0.0
        A[0, dim] = 75.02750896
        for i in range(1, dim):
            A[0, i] = 0.0
            A[0, i + dim] = 100.0

        tree = PRTree(np.array([0], dtype=np.int64), A)

        # Insert with float64 (small gap)
        B = np.zeros(2 * dim, dtype=np.float64)
        B[0] = 75.02751435
        B[dim] = 100.0
        for i in range(1, dim):
            B[i] = 0.0
            B[i + dim] = 100.0

        tree.insert(idx=1, bb=B)

        # Query should not find intersection due to small gap
        result = tree.query(B)
        assert 0 not in result, "Should not find item 0 due to small gap with float64 precision"
        assert 1 in result, "Should find item 1 (self)"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_float32_loses_precision(self, PRTree, dim):
        """Verify that float32 insert may lose precision for small gaps."""
        # Create tree with float64 construction
        A = np.zeros((1, 2 * dim), dtype=np.float64)
        A[0, 0] = 0.0
        A[0, dim] = 75.02750896
        for i in range(1, dim):
            A[0, i] = 0.0
            A[0, i + dim] = 100.0

        tree = PRTree(np.array([0], dtype=np.int64), A)

        # Insert with float32 (small gap, may cause false positive)
        B = np.zeros(2 * dim, dtype=np.float32)
        B[0] = 75.02751435
        B[dim] = 100.0
        for i in range(1, dim):
            B[i] = 0.0
            B[i + dim] = 100.0

        tree.insert(idx=1, bb=B)

        # Query - item 1 won't have exact coordinates, so refinement won't apply to it
        result = tree.query(B)
        assert 1 in result, "Should find item 1 (self)"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_rebuild_preserves_idx2exact(self, PRTree, dim):
        """Verify that rebuild() preserves idx2exact for precision."""
        # Create tree with float64 to populate idx2exact
        n = 10
        idx = np.arange(n, dtype=np.int64)
        boxes = np.random.rand(n, 2 * dim) * 100
        boxes = boxes.astype(np.float64)
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Insert more items to trigger rebuild
        for i in range(n, n + 100):
            box = np.random.rand(2 * dim) * 100
            box = box.astype(np.float64)
            for d in range(dim):
                box[d + dim] += box[d] + 1
            tree.insert(idx=i, bb=box)

        # Create a small-gap query that should only work with float64 refinement
        # Query box is to the right of boxes[0] with a small gap
        query = np.zeros(2 * dim, dtype=np.float64)
        query[0] = boxes[0, dim] + 1e-6  # Small gap after original box's max
        query[dim] = boxes[0, dim] + 10.0  # Query max
        for i in range(1, dim):
            # Overlap in other dimensions
            query[i] = boxes[0, i] - 10
            query[i + dim] = boxes[0, i + dim] + 10

        result = tree.query(query)
        # Should not find item 0 if idx2exact is preserved and working
        # The gap of 1e-6 should be detected with float64 precision
        assert 0 not in result, "Should not find item 0 due to small gap (idx2exact should be preserved after rebuild)"
