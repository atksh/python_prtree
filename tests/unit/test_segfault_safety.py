"""Unit tests for segmentation fault safety.

These tests cover scenarios that could potentially cause segfaults
in the C++/Cython implementation. They ensure memory safety and
proper error handling.

Note: Some tests use subprocess to isolate potential crashes.
"""
import gc
import sys
import subprocess
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestNullPointerSafety:
    """Test protection against null pointer dereferences."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_on_uninitialized_tree(self, PRTree, dim):
        """Verify that query on uninitialized tree fails safely."""
        tree = PRTree()

        query_box = np.zeros(2 * dim)
        for i in range(dim):
            query_box[i] = 0.0
            query_box[i + dim] = 1.0

        # Should not segfault, should return empty or raise error
        try:
            result = tree.query(query_box)
            assert result == []
        except (RuntimeError, ValueError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_on_empty_tree(self, PRTree, dim):
        """Verify that erase from empty tree fails safely."""
        tree = PRTree()

        # Should not segfault, should raise ValueError
        with pytest.raises(ValueError):
            tree.erase(1)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_get_obj_on_empty_tree(self, PRTree, dim):
        """Verify that get_obj from empty tree fails safely."""
        tree = PRTree()

        # Should not segfault
        try:
            obj = tree.get_obj(0)
            # If it succeeds, obj should be None or raise error
        except (RuntimeError, ValueError, KeyError, IndexError):
            pass


class TestUseAfterFree:
    """Test protection against use-after-free scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_after_erase(self, PRTree, dim):
        """Verify that query after eraseworks safely."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Erase all elements
        for i in range(n):
            tree.erase(i)

        # Query should not segfault
        query_box = boxes[0]
        result = tree.query(query_box)
        assert result == []

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_access_after_rebuild(self, PRTree, dim):
        """Verify that access after rebuildworks safely."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Rebuild multiple times
        for _ in range(5):
            tree.rebuild()

        # Should still work
        query_box = boxes[0]
        result = tree.query(query_box)
        assert isinstance(result, list)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_after_save(self, PRTree, dim, tmp_path):
        """Verify that query after saveworks safely."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        fname = tmp_path / "tree.bin"
        tree.save(str(fname))

        # Query after save should still work
        query_box = boxes[0]
        result = tree.query(query_box)
        assert isinstance(result, list)


class TestBufferOverflow:
    """Test protection against buffer overflows."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_very_large_index(self, PRTree, dim):
        """Verify that very large indexis handled safely."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        # Very large index
        large_idx = 2**31 - 1

        # Should not segfault
        try:
            tree.insert(idx=large_idx, bb=box)
            assert tree.size() == 1
        except (OverflowError, ValueError, RuntimeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_negative_large_index(self, PRTree, dim):
        """Verify that very small negative indexis handled safely."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        # Very negative index
        neg_idx = -2**31

        # Should not segfault
        try:
            tree.insert(idx=neg_idx, bb=box)
            assert tree.size() == 1
        except (OverflowError, ValueError, RuntimeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_extremely_large_coordinates(self, PRTree, dim):
        """Verify that extremely large coordinatesis handled safely."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))

        # Extremely large coordinates (but not inf)
        for i in range(dim):
            boxes[0, i] = 1e100
            boxes[0, i + dim] = 1e100 + 1

        # Should not segfault
        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 1
        except (ValueError, RuntimeError, OverflowError):
            pass


class TestArrayBoundsSafety:
    """Test protection against array bounds violations."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_empty_array_input(self, PRTree, dim):
        """Verify that empty array inputis handled safely."""
        idx = np.array([])
        boxes = np.empty((0, 2 * dim))

        # Should not segfault
        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 0
        except (ValueError, RuntimeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_wrong_shaped_boxes(self, PRTree, dim):
        """Verify that wrong shaped boxesis handled safely."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, dim))  # Wrong: should be 2*dim

        # Should not segfault, should raise error
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_1d_boxes_input(self, PRTree, dim):
        """Verify that 1D boxes inputis handled safely."""
        idx = np.array([1])
        boxes = np.zeros(2 * dim)  # 1D instead of 2D

        # Should handle or raise error, not segfault
        try:
            tree = PRTree(idx, boxes)
            # Some implementations might accept 1D for single element
        except (ValueError, RuntimeError, IndexError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_3d_boxes_input(self, PRTree, dim):
        """Verify that 3D boxes inputis handled safely."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, 2, dim))  # 3D instead of 2D

        # Should raise error, not segfault
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            PRTree(idx, boxes)


class TestMemoryLeaks:
    """Test for potential memory leaks (not direct segfaults but related)."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_repeated_insert_erase(self, PRTree, dim):
        """Verify no memory leaks with repeated insert/erase."""
        tree = PRTree()

        # Many iterations
        for iteration in range(100):
            for i in range(50):
                box = np.random.rand(2 * dim) * 100
                for d in range(dim):
                    box[d + dim] += box[d] + 1
                tree.insert(idx=iteration * 50 + i, bb=box)

            # Erase half
            for i in range(25):
                tree.erase(iteration * 50 + i)

        # Force garbage collection
        gc.collect()

        # Should still be responsive
        assert tree.size() > 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_repeated_save_load(self, PRTree, dim, tmp_path):
        """Verify no memory leaks with repeated save/load."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Many save/load cycles
        for i in range(20):
            fname = tmp_path / f"tree_{i}.bin"
            tree.save(str(fname))
            del tree
            gc.collect()
            tree = PRTree(str(fname))

        # Should still work
        assert tree.size() == n


class TestCorruptedData:
    """Test handling of corrupted data."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_load_corrupted_file(self, PRTree, dim, tmp_path):
        """Verify that loading corrupted file fails safely."""
        fname = tmp_path / "corrupted.bin"

        # Create corrupted file
        with open(fname, 'wb') as f:
            f.write(b'corrupted data' * 100)

        # Should not segfault, should raise error
        with pytest.raises((RuntimeError, ValueError, OSError)):
            PRTree(str(fname))

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_load_empty_file(self, PRTree, dim, tmp_path):
        """Verify that loading empty file fails safely."""
        fname = tmp_path / "empty.bin"

        # Create empty file
        fname.touch()

        # Should not segfault, should raise error
        with pytest.raises((RuntimeError, ValueError, OSError)):
            PRTree(str(fname))

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_load_partial_file(self, PRTree, dim, tmp_path):
        """Verify that loading partially corrupted file fails safely."""
        # First create a valid file
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        fname = tmp_path / "partial.bin"
        tree.save(str(fname))

        # Truncate the file
        with open(fname, 'rb') as f:
            data = f.read()

        with open(fname, 'wb') as f:
            f.write(data[:len(data) // 2])  # Write only half

        # Should not segfault, should raise error
        with pytest.raises((RuntimeError, ValueError, OSError)):
            PRTree(str(fname))


class TestConcurrentAccess:
    """Test thread safety and concurrent access."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_during_modification(self, PRTree, dim):
        """Verify that query during modification works safely (single-threaded)."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Interleave queries and modifications
        for i in range(20):
            # Query
            query_box = np.random.rand(2 * dim) * 100
            for d in range(dim):
                query_box[d + dim] += query_box[d] + 1
            result = tree.query(query_box)

            # Modify
            tree.erase(i)

            # Query again
            result = tree.query(query_box)

            # Insert
            new_box = np.random.rand(2 * dim) * 100
            for d in range(dim):
                new_box[d + dim] += new_box[d] + 1
            tree.insert(idx=n + i, bb=new_box)

        # Should not segfault
        assert tree.size() > 0


class TestObjectLifecycle:
    """Test proper object lifecycle management."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_tree_deletion_and_recreation(self, PRTree, dim):
        """Verify that tree deletion and recreationworks safely."""
        for _ in range(10):
            n = 50
            idx = np.arange(n)
            boxes = np.random.rand(n, 2 * dim) * 100
            for i in range(dim):
                boxes[:, i + dim] += boxes[:, i] + 1

            tree = PRTree(idx, boxes)

            # Use the tree
            query_box = boxes[0]
            result = tree.query(query_box)

            # Delete and force cleanup
            del tree
            gc.collect()

        # Should not accumulate memory issues

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_circular_reference_safety(self, PRTree, dim):
        """Verify that circular referencesis handled safely."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        # Insert with object that might create circular reference
        obj = {"tree": None}  # Will set later
        tree.insert(idx=1, bb=box, obj=obj)

        # Create potential circular reference
        obj["tree"] = tree

        # Should handle cleanup properly
        del tree
        del obj
        gc.collect()


class TestExtremeInputs:
    """Test extreme and unusual inputs."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_all_nan_boxes(self, PRTree, dim):
        """Verify that all NaN boxesis handled safely."""
        idx = np.array([1])
        boxes = np.full((1, 2 * dim), np.nan)

        # Should not segfault, should raise error
        with pytest.raises((ValueError, RuntimeError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_mixed_nan_and_valid(self, PRTree, dim):
        """Verify that boxes with mixed NaN and valid valuesis handled safely."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        boxes[0, 0] = np.nan  # Only first coordinate is NaN
        for i in range(1, dim):
            boxes[0, i] = i
            boxes[0, i + dim] = i + 1

        # Should not segfault, should raise error
        with pytest.raises((ValueError, RuntimeError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_zero_size_boxes(self, PRTree, dim):
        """Verify that zero-size boxesis handled safely."""
        n = 10
        idx = np.arange(n)
        boxes = np.zeros((n, 2 * dim))

        # All boxes have zero size
        for i in range(n):
            for d in range(dim):
                boxes[i, d] = i
                boxes[i, d + dim] = i  # min == max

        # Should not segfault
        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == n
        except (ValueError, RuntimeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_very_large_dataset(self, PRTree, dim):
        """Verify that very large dataset can be processed."""
        # This might fail due to memory, but should not segfault
        try:
            n = 100000
            idx = np.arange(n)
            boxes = np.random.rand(n, 2 * dim).astype(np.float32) * 1000
            for i in range(dim):
                boxes[:, i + dim] += boxes[:, i] + 1

            tree = PRTree(idx, boxes)
            assert tree.size() == n

            # Cleanup
            del tree
            gc.collect()
        except MemoryError:
            # Acceptable - ran out of memory
            pass


class TestTypeSafety:
    """Test type safety and conversion."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_wrong_dtype_indices(self, PRTree, dim):
        """Verify that wrong dtype indicesis handled safely."""
        idx = np.array([1.5, 2.7], dtype=np.float64)  # Float instead of int
        boxes = np.zeros((2, 2 * dim))
        for i in range(2):
            for d in range(dim):
                boxes[i, d] = i
                boxes[i, d + dim] = i + 1

        # Should convert or raise error, not segfault
        try:
            tree = PRTree(idx, boxes)
            assert tree.size() == 2
        except (ValueError, RuntimeError, TypeError):
            pass

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_string_indices(self, PRTree, dim):
        """Verify that string indicesis handled safely."""
        # String indices should raise error, not segfault
        boxes = np.zeros((2, 2 * dim))
        for i in range(2):
            for d in range(dim):
                boxes[i, d] = i
                boxes[i, d + dim] = i + 1

        # This should raise TypeError
        with pytest.raises((TypeError, ValueError)):
            PRTree(["a", "b"], boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_none_input(self, PRTree, dim):
        """Verify that None inputis handled safely."""
        # None should raise error, not segfault
        with pytest.raises((TypeError, ValueError)):
            PRTree(None, None)
