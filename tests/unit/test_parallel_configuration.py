"""Tests for parallel configuration and thread count settings.

Tests verify that batch_query parallelization behaves correctly with:
- Different thread counts (if configurable)
- Different dataset sizes
- Different query batch sizes

Note: The library uses C++ std::thread for batch_query parallelization.
This test suite verifies correct behavior across different configurations.
"""
import os
import time
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestParallelScaling:
    """Test parallel performance scaling."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    @pytest.mark.parametrize("query_count", [10, 100, 1000])
    def test_batch_query_scaling(self, PRTree, dim, query_count):
        """batch_queryが異なるクエリ数で正しく動作することを確認."""
        np.random.seed(42)
        n = 1000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        queries = np.random.rand(query_count, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        # Run batch query
        start_time = time.time()
        results = tree.batch_query(queries)
        elapsed = time.time() - start_time

        # Verify correctness
        assert len(results) == query_count
        for result in results:
            assert isinstance(result, list)

        print(f"batch_query({query_count} queries) took {elapsed:.4f}s")

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    @pytest.mark.parametrize("tree_size", [100, 1000, 10000])
    def test_batch_query_tree_size_scaling(self, PRTree, dim, tree_size):
        """異なるツリーサイズでbatch_queryが正しく動作することを確認."""
        np.random.seed(42)
        idx = np.arange(tree_size)
        boxes = np.random.rand(tree_size, 2 * dim).astype(np.float32) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        queries = np.random.rand(100, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        results = tree.batch_query(queries)

        assert len(results) == 100
        for result in results:
            assert isinstance(result, list)


class TestBatchVsSingleQuery:
    """Test batch_query vs individual query consistency."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 500])
    def test_batch_query_consistency(self, PRTree, dim, batch_size):
        """batch_queryと個別queryの結果が一致することを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        queries = np.random.rand(batch_size, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        # Batch query
        batch_results = tree.batch_query(queries)

        # Individual queries
        individual_results = [tree.query(queries[i]) for i in range(batch_size)]

        # Compare
        assert len(batch_results) == len(individual_results)
        for i in range(batch_size):
            assert set(batch_results[i]) == set(individual_results[i]), \
                f"Mismatch at query {i}: batch={batch_results[i]}, individual={individual_results[i]}"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_batch_query_performance_benefit(self, PRTree, dim):
        """batch_queryが個別queryより速いことを確認（目安）."""
        np.random.seed(42)
        n = 2000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float32) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        n_queries = 500
        queries = np.random.rand(n_queries, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        # Batch query
        start = time.time()
        batch_results = tree.batch_query(queries)
        batch_time = time.time() - start

        # Individual queries
        start = time.time()
        individual_results = [tree.query(queries[i]) for i in range(n_queries)]
        individual_time = time.time() - start

        print(f"Batch: {batch_time:.4f}s, Individual: {individual_time:.4f}s, " +
              f"Speedup: {individual_time/batch_time:.2f}x")

        # Verify correctness
        for i in range(n_queries):
            assert set(batch_results[i]) == set(individual_results[i])

        # Batch should generally be faster for large query counts
        # (but we don't enforce this as it depends on hardware)


class TestParallelCorrectness:
    """Test correctness of parallel execution."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_deterministic(self, PRTree, dim):
        """batch_queryが決定的な結果を返すことを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        queries = np.random.rand(100, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        # Run multiple times
        results1 = tree.batch_query(queries)
        results2 = tree.batch_query(queries)
        results3 = tree.batch_query(queries)

        # Should be identical
        for i in range(100):
            assert set(results1[i]) == set(results2[i]) == set(results3[i])

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_no_data_races(self, PRTree, dim):
        """batch_queryでデータ競合がないことを確認（正しい結果が返る）."""
        np.random.seed(42)
        n = 1000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Large batch to stress parallel execution
        n_queries = 1000
        queries = np.random.rand(n_queries, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        batch_results = tree.batch_query(queries)

        # Verify each result is correct
        for i in range(n_queries):
            expected = tree.query(queries[i])
            assert set(batch_results[i]) == set(expected), \
                f"Data race detected at query {i}"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_batch_query_with_duplicates(self, PRTree, dim):
        """重複クエリでbatch_queryが正しく動作することを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Create queries with duplicates
        query1 = np.random.rand(2 * dim) * 100
        for i in range(dim):
            query1[i + dim] += query1[i] + 1

        queries = np.tile(query1, (100, 1))  # 100 identical queries

        results = tree.batch_query(queries)

        # All results should be identical
        assert len(results) == 100
        first_result_set = set(results[0])
        for result in results:
            assert set(result) == first_result_set


class TestEdgeCasesParallel:
    """Test edge cases in parallel execution."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_single_query(self, PRTree, dim):
        """1つのクエリでbatch_queryが正しく動作することを確認."""
        np.random.seed(42)
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        query = np.random.rand(1, 2 * dim) * 100
        for i in range(dim):
            query[:, i + dim] += query[:, i] + 1

        batch_result = tree.batch_query(query)
        single_result = tree.query(query[0])

        assert len(batch_result) == 1
        assert set(batch_result[0]) == set(single_result)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_empty_tree(self, PRTree, dim):
        """空のツリーでbatch_queryが正しく動作することを確認."""
        tree = PRTree()

        queries = np.random.rand(50, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        results = tree.batch_query(queries)

        assert len(results) == 50
        for result in results:
            assert result == []

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_single_element_tree(self, PRTree, dim):
        """1要素のツリーでbatch_queryが正しく動作することを確認."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = 0.0
            boxes[0, i + dim] = 1.0

        tree = PRTree(idx, boxes)

        queries = np.random.rand(50, 2 * dim) * 2  # Some will intersect
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 0.1

        results = tree.batch_query(queries)

        assert len(results) == 50
        for i, result in enumerate(results):
            # Verify correctness
            expected = tree.query(queries[i])
            assert set(result) == set(expected)


class TestQueryIntersectionsParallel:
    """Test query_intersections which may also use parallelization."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    @pytest.mark.parametrize("tree_size", [50, 200, 500])
    def test_query_intersections_scaling(self, PRTree, dim, tree_size):
        """異なるツリーサイズでquery_intersectionsが正しく動作することを確認."""
        np.random.seed(42)
        idx = np.arange(tree_size)
        boxes = np.random.rand(tree_size, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 5  # Make boxes overlap

        tree = PRTree(idx, boxes)

        start = time.time()
        pairs = tree.query_intersections()
        elapsed = time.time() - start

        # Verify output
        assert pairs.ndim == 2
        assert pairs.shape[1] == 2
        if pairs.shape[0] > 0:
            assert np.all(pairs[:, 0] < pairs[:, 1])

        print(f"query_intersections({tree_size} boxes) found {pairs.shape[0]} pairs in {elapsed:.4f}s")

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_query_intersections_deterministic(self, PRTree, dim):
        """query_intersectionsが決定的な結果を返すことを確認."""
        np.random.seed(42)
        n = 200
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 3

        tree = PRTree(idx, boxes)

        # Run multiple times
        pairs1 = tree.query_intersections()
        pairs2 = tree.query_intersections()
        pairs3 = tree.query_intersections()

        # Should be identical
        assert np.array_equal(pairs1, pairs2)
        assert np.array_equal(pairs2, pairs3)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_query_intersections_correctness(self, PRTree, dim):
        """query_intersectionsの結果が正しいことを確認（並列化の検証）."""
        np.random.seed(42)
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 2

        tree = PRTree(idx, boxes)

        pairs = tree.query_intersections()

        # Verify each pair actually intersects
        def has_intersect(x, y, dim):
            return all([max(x[i], y[i]) <= min(x[i + dim], y[i + dim]) for i in range(dim)])

        for pair in pairs:
            i, j = pair
            assert has_intersect(boxes[i], boxes[j], dim), \
                f"Pair ({i}, {j}) reported as intersecting but doesn't"

        # Verify no pairs are missing (naive check)
        expected_pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                if has_intersect(boxes[i], boxes[j], dim):
                    expected_pairs.add((i, j))

        actual_pairs = set(map(tuple, pairs))
        assert actual_pairs == expected_pairs, \
            f"Missing pairs: {expected_pairs - actual_pairs}, Extra pairs: {actual_pairs - expected_pairs}"


class TestRebuildParallel:
    """Test rebuild in parallel scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_rebuild_after_parallel_queries(self, PRTree, dim):
        """並列クエリ後のrebuildが正しく動作することを確認."""
        np.random.seed(42)
        n = 500
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Run many batch queries
        for _ in range(10):
            queries = np.random.rand(100, 2 * dim) * 100
            for i in range(dim):
                queries[:, i + dim] += queries[:, i] + 1
            tree.batch_query(queries)

        # Rebuild
        tree.rebuild()

        # Verify still works
        queries = np.random.rand(50, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        results = tree.batch_query(queries)
        assert len(results) == 50
