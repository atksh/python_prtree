"""Unit tests for PRTree save/load operations."""
import gc
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestNormalPersistence:
    """Test normal save/load scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_save_and_load(self, PRTree, dim, tmp_path):
        """Verify that save and loadworks."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        fname = tmp_path / "tree.bin"
        tree.save(str(fname))

        # Load via constructor
        loaded_tree = PRTree(str(fname))
        assert loaded_tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_load_via_load_method(self, PRTree, dim, tmp_path):
        """Verify that load via load methodworks."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        fname = tmp_path / "tree.bin"
        tree.save(str(fname))

        # Load via load() method
        new_tree = PRTree()
        new_tree.load(str(fname))
        assert new_tree.size() == n


class TestErrorPersistence:
    """Test save/load with invalid inputs."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_load_non_existent_file(self, PRTree, dim):
        """Verify that loading non-existent fileraises an error."""
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            PRTree("/non/existent/path/tree.bin")

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_save_to_invalid_path(self, PRTree, dim):
        """Verify that save to invalid pathraises an error."""
        tree = PRTree()
        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0
        tree.insert(idx=1, bb=box)

        with pytest.raises((OSError, RuntimeError)):
            tree.save("/non/existent/directory/tree.bin")


class TestConsistencyPersistence:
    """Test save/load consistency."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_results_after_save_load(self, PRTree, dim, tmp_path):
        """Verify that query results after save/loadmatches."""
        np.random.seed(42)
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Query before save
        query_box = np.random.rand(2 * dim) * 100
        for i in range(dim):
            query_box[i + dim] += query_box[i] + 1

        result_before = tree.query(query_box)

        # Save and load
        fname = tmp_path / "tree.bin"
        tree.save(str(fname))
        loaded_tree = PRTree(str(fname))

        # Query after load
        result_after = loaded_tree.query(query_box)

        assert set(result_before) == set(result_after)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_float64_precision_after_save_load(self, PRTree, dim, tmp_path):
        """Verify that float64 precision is preserved after save/load."""
        A = np.zeros((1, 2 * dim), dtype=np.float64)
        B = np.zeros((1, 2 * dim), dtype=np.float64)

        # Small gap in first dimension
        A[0, 0] = 0.0
        A[0, dim] = 75.02750896
        B[0, 0] = 75.02751435
        B[0, dim] = 100.0

        # Fill other dimensions
        for i in range(1, dim):
            A[0, i] = 0.0
            A[0, i + dim] = 100.0
            B[0, i] = 0.0
            B[0, i + dim] = 100.0

        tree = PRTree(np.array([0]), A)

        # Query before save
        result_before = tree.query(B[0])

        # Save and load
        fname = tmp_path / "tree.bin"
        tree.save(str(fname))

        del tree
        gc.collect()

        loaded_tree = PRTree(str(fname))

        # Query after load
        result_after = loaded_tree.query(B[0])

        # Should match (no intersection)
        assert result_before == result_after == []

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_multiple_save_load_cycles(self, PRTree, dim, tmp_path):
        """Verify that results across multiple save/load cyclesmatches."""
        np.random.seed(42)
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float64) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1e-5

        tree = PRTree(idx, boxes)

        queries = np.random.rand(10, 2 * dim).astype(np.float64) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1e-5

        results_initial = tree.batch_query(queries)

        # First save/load cycle
        fname1 = tmp_path / "tree1.bin"
        tree.save(str(fname1))
        del tree
        gc.collect()

        tree1 = PRTree(str(fname1))
        results1 = tree1.batch_query(queries)

        # Second save/load cycle
        fname2 = tmp_path / "tree2.bin"
        tree1.save(str(fname2))
        del tree1
        gc.collect()

        tree2 = PRTree(str(fname2))
        results2 = tree2.batch_query(queries)

        # All results should match
        assert results_initial == results1 == results2
