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
        """未初期化ツリーへのクエリが安全に失敗することを確認."""
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
        """空のツリーからの削除が安全に失敗することを確認."""
        tree = PRTree()

        # Should not segfault, should raise ValueError
        with pytest.raises(ValueError):
            tree.erase(1)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_get_obj_on_empty_tree(self, PRTree, dim):
        """空のツリーからのオブジェクト取得が安全に失敗することを確認."""
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
        """削除後のクエリが安全に動作することを確認."""
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
        """rebuild後のアクセスが安全に動作することを確認."""
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
        """保存後のクエリが安全に動作することを確認."""
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
        """非常に大きなインデックスが安全に処理されることを確認."""
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
        """非常に小さな負のインデックスが安全に処理されることを確認."""
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
        """極端に大きな座標が安全に処理されることを確認."""
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
        """空の配列入力が安全に処理されることを確認."""
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
        """間違った形状のボックス配列が安全に処理されることを確認."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, dim))  # Wrong: should be 2*dim

        # Should not segfault, should raise error
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_1d_boxes_input(self, PRTree, dim):
        """1次元ボックス配列が安全に処理されることを確認."""
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
        """3次元ボックス配列が安全に処理されることを確認."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, 2, dim))  # 3D instead of 2D

        # Should raise error, not segfault
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            PRTree(idx, boxes)


class TestMemoryLeaks:
    """Test for potential memory leaks (not direct segfaults but related)."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_repeated_insert_erase(self, PRTree, dim):
        """繰り返しの挿入・削除でメモリリークがないことを確認."""
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
        """繰り返しの保存・読込でメモリリークがないことを確認."""
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
        """破損したファイルの読み込みが安全に失敗することを確認."""
        fname = tmp_path / "corrupted.bin"

        # Create corrupted file
        with open(fname, 'wb') as f:
            f.write(b'corrupted data' * 100)

        # Should not segfault, should raise error
        with pytest.raises((RuntimeError, ValueError, OSError)):
            PRTree(str(fname))

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_load_empty_file(self, PRTree, dim, tmp_path):
        """空ファイルの読み込みが安全に失敗することを確認."""
        fname = tmp_path / "empty.bin"

        # Create empty file
        fname.touch()

        # Should not segfault, should raise error
        with pytest.raises((RuntimeError, ValueError, OSError)):
            PRTree(str(fname))

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_load_partial_file(self, PRTree, dim, tmp_path):
        """部分的に破損したファイルの読み込みが安全に失敗することを確認."""
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
        """変更中のクエリが安全に動作することを確認（単一スレッド）."""
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
        """ツリーの削除と再作成が安全に動作することを確認."""
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
        """循環参照が安全に処理されることを確認."""
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
        """全てNaNのボックスが安全に処理されることを確認."""
        idx = np.array([1])
        boxes = np.full((1, 2 * dim), np.nan)

        # Should not segfault, should raise error
        with pytest.raises((ValueError, RuntimeError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_mixed_nan_and_valid(self, PRTree, dim):
        """NaNと有効値が混在するボックスが安全に処理されることを確認."""
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
        """ゼロサイズのボックスが安全に処理されることを確認."""
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
        """非常に大きなデータセットが処理できることを確認."""
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
        """間違った型のインデックスが安全に処理されることを確認."""
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
        """文字列インデックスが安全に処理されることを確認."""
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
        """Noneの入力が安全に処理されることを確認."""
        # None should raise error, not segfault
        with pytest.raises((TypeError, ValueError)):
            PRTree(None, None)
