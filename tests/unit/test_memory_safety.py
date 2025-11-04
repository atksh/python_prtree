"""Memory safety and bounds checking tests.

These tests verify that the library properly validates inputs and
handles edge cases related to memory management without causing
segmentation faults or memory corruption.
"""
import gc
import sys
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestInputValidation:
    """Test input validation to prevent memory issues."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_negative_box_dimensions(self, PRTree, dim):
        """Verify that negative box dimensions are properly rejected."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))

        # Set min > max (invalid)
        for i in range(dim):
            boxes[0, i] = 100.0
            boxes[0, i + dim] = 0.0

        with pytest.raises((ValueError, RuntimeError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_misaligned_array(self, PRTree, dim):
        """Verify that misaligned arrayis handled safely."""
        # Create non-contiguous array
        idx = np.arange(10)
        boxes_full = np.random.rand(20, 2 * dim) * 100
        for i in range(dim):
            boxes_full[:, i + dim] += boxes_full[:, i] + 1

        # Take every other row (non-contiguous)
        boxes = boxes_full[::2, :]
        assert not boxes.flags['C_CONTIGUOUS']

        # Should handle or raise error, not crash
        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 10
        except (ValueError, RuntimeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_fortran_order_array(self, PRTree, dim):
        """Verify that Fortran order arrayis handled safely."""
        idx = np.arange(10)
        boxes = np.asfortranarray(np.random.rand(10, 2 * dim) * 100)
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        assert boxes.flags['F_CONTIGUOUS']

        # Should handle or raise error, not crash
        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 10
        except (ValueError, RuntimeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_readonly_array(self, PRTree, dim):
        """Verify that readonly arrayis handled safely."""
        idx = np.arange(10)
        boxes = np.random.rand(10, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        boxes.flags.writeable = False

        # Should handle read-only arrays
        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 10
        except (ValueError, RuntimeError):
            pass


class TestMemoryBounds:
    """Test memory bounds checking."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_out_of_bounds_index_access(self, PRTree, dim):
        """Verify that out-of-bounds index accessis handled safely."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Try to access object with out-of-bounds index
        try:
            obj = tree.get_obj(999)
        except (ValueError, RuntimeError, KeyError, IndexError):
            pass

        # Try to erase out-of-bounds index
        try:
            tree.erase(999)
        except (ValueError, RuntimeError, KeyError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_with_wrong_size_array(self, PRTree, dim):
        """Verify that query with wrong size array is handled safely."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Too small
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            tree.query(np.zeros(2 * dim + 1))  # Wrong size (one extra dimension)

        # Too large
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            tree.query(np.zeros(3 * dim))

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_inconsistent_shapes(self, PRTree, dim):
        """Verify that batch_query with inconsistent shapes is handled safely."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Wrong second dimension
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            queries = np.zeros((5, 2 * dim + 1))  # Wrong size
            tree.batch_query(queries)


class TestGarbageCollection:
    """Test interaction with Python's garbage collector."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_tree_gc_cycle(self, PRTree, dim):
        """Verify that tree deletion during garbage collection cycle is safeVerify that."""
        for _ in range(10):
            idx = np.arange(100)
            boxes = np.random.rand(100, 2 * dim) * 100
            for i in range(dim):
                boxes[:, i + dim] += boxes[:, i] + 1

            tree = PRTree(idx, boxes)

            # Use the tree
            query_box = boxes[0]
            result = tree.query(query_box)

            # Trigger GC while tree is in scope
            gc.collect()

            # Use again
            result = tree.query(query_box)

            # Delete and force GC
            del tree
            gc.collect()

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_numpy_array_lifecycle(self, PRTree, dim):
        """Verify that numpy array lifecycle is managed correctly."""
        idx = np.arange(100)
        boxes = np.random.rand(100, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        # Keep reference to original boxes
        original_boxes = boxes.copy()

        tree = PRTree(idx, boxes)

        # Delete original arrays
        del idx
        del boxes
        gc.collect()

        # Tree should still work
        query_box = original_boxes[0]
        result = tree.query(query_box)
        assert isinstance(result, list)


class TestEdgeCaseArrays:
    """Test edge case array configurations."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_single_precision_underflow(self, PRTree, dim):
        """Verify that float32 underflowis handled safely."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim), dtype=np.float32)

        # Very small numbers that might underflow in float32
        for i in range(dim):
            boxes[0, i] = 1e-40
            boxes[0, i + dim] = 1e-40 + 1e-41

        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 1
        except (ValueError, RuntimeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_subnormal_numbers(self, PRTree, dim):
        """Verify that subnormal numbersis handled safely."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim), dtype=np.float64)

        # Subnormal numbers
        for i in range(dim):
            boxes[0, i] = sys.float_info.min / 2
            boxes[0, i + dim] = sys.float_info.min

        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 1

            # Query with subnormal
            query_box = boxes[0].copy()
            result = tree.query(query_box)
        except (ValueError, RuntimeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_mixed_special_values(self, PRTree, dim):
        """Verify handling of mixed special values."""
        idx = np.array([1, 2, 3])
        boxes = np.zeros((3, 2 * dim))

        # Box 1: Normal values
        for i in range(dim):
            boxes[0, i] = 0.0
            boxes[0, i + dim] = 1.0

        # Box 2: Very large values
        for i in range(dim):
            boxes[1, i] = 1e100
            boxes[1, i + dim] = 1e101

        # Box 3: Very small values
        for i in range(dim):
            boxes[2, i] = 1e-100
            boxes[2, i + dim] = 1e-99

        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 3
        except (ValueError, RuntimeError):
            pass


class TestConcurrentModification:
    """Test protection against concurrent modification (single-threaded)."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_modify_during_batch_query(self, PRTree, dim):
        """Verify that modifications during batch_query are safe (implementation-dependent)."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        queries = np.random.rand(50, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        # This should complete without crash
        # (implementation might use snapshot or raise error)
        try:
            result = tree.batch_query(queries)
            assert len(result) == 50
        except RuntimeError:
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_during_iteration(self, PRTree, dim):
        """Verify that insert during iteration is safe."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Query and insert in interleaved manner
        for i in range(20):
            query_box = boxes[i % n]
            result = tree.query(query_box)

            new_box = np.random.rand(2 * dim) * 100
            for d in range(dim):
                new_box[d + dim] += new_box[d] + 1
            tree.insert(idx=n + i, bb=new_box)

        # Should complete without crash
        assert tree.size() > n


class TestResourceExhaustion:
    """Test behavior under resource exhaustion."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_many_small_insertions(self, PRTree, dim):
        """Verify that many small insertions can be processed."""
        tree = PRTree()

        # Many small insertions
        for i in range(10000):
            box = np.random.rand(2 * dim) * 1000
            for d in range(dim):
                box[d + dim] += box[d] + 1

            tree.insert(idx=i, bb=box)

            # Periodically query to ensure tree stays consistent
            if i % 1000 == 0:
                result = tree.query(box)
                assert i in result

        assert tree.size() == 10000

        # Cleanup
        del tree
        gc.collect()

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2)])  # Only 2D to save time
    def test_large_single_tree(self, PRTree, dim):
        """Verify that large single tree can be processed."""
        try:
            n = 50000
            idx = np.arange(n)
            boxes = np.random.rand(n, 2 * dim).astype(np.float32) * 1000
            for i in range(dim):
                boxes[:, i + dim] += boxes[:, i] + 1

            tree = PRTree(idx, boxes)
            assert tree.size() == n

            # Sample query
            query_box = boxes[0]
            result = tree.query(query_box)
            assert isinstance(result, list)

            # Cleanup
            del tree
            del boxes
            gc.collect()
        except MemoryError:
            pytest.skip("Not enough memory for this test")


class TestNumpyDtypes:
    """Test various numpy data types."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_int32_indices(self, PRTree, dim):
        """Verify that int32 indices can be processed."""
        idx = np.arange(10, dtype=np.int32)
        boxes = np.random.rand(10, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == 10

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_int64_indices(self, PRTree, dim):
        """Verify that int64 indices can be processed."""
        idx = np.arange(10, dtype=np.int64)
        boxes = np.random.rand(10, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == 10

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_uint_indices(self, PRTree, dim):
        """Verify that unsigned int indices can be processed."""
        idx = np.arange(10, dtype=np.uint32)
        boxes = np.random.rand(10, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 10
        except (ValueError, RuntimeError, TypeError):
            # Unsigned might not be supported
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_float16_boxes(self, PRTree, dim):
        """Verify that float16 boxes can be processed (or error)."""
        idx = np.arange(10)
        boxes = np.random.rand(10, 2 * dim).astype(np.float16) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 10
        except (ValueError, RuntimeError, TypeError):
            # float16 might not be supported
            pass
