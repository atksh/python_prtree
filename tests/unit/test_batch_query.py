"""Unit tests for PRTree batch_query operations."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


def has_intersect(x, y, dim):
    """Helper function to check if two boxes intersect."""
    return all([max(x[i], y[i]) <= min(x[i + dim], y[i + dim]) for i in range(dim)])


class TestNormalBatchQuery:
    """Test normal batch query scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_returns_correct_results(self, PRTree, dim):
        """バッチクエリが正しい結果を返すことを確認."""
        np.random.seed(42)
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Batch query
        n_queries = 10
        queries = np.random.rand(n_queries, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        results = tree.batch_query(queries)

        assert len(results) == n_queries
        for i, result in enumerate(results):
            expected = [idx[j] for j in range(n) if has_intersect(boxes[j], queries[i], dim)]
            assert set(result) == set(expected)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_empty_queries(self, PRTree, dim):
        """空のクエリ配列でバッチクエリが動作することを確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Empty query array
        queries = np.empty((0, 2 * dim))
        results = tree.batch_query(queries)

        assert len(results) == 0


class TestConsistencyBatchQuery:
    """Test batch query consistency with single query."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_vs_query_consistency(self, PRTree, dim):
        """batch_queryとqueryの結果が一致することを確認."""
        np.random.seed(42)
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        n_queries = 20
        queries = np.random.rand(n_queries, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        batch_results = tree.batch_query(queries)

        for i, query in enumerate(queries):
            single_result = tree.query(query)
            assert set(batch_results[i]) == set(single_result)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_single_query_as_batch(self, PRTree, dim):
        """1つのクエリをバッチとして実行した場合の動作確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        query = np.random.rand(2 * dim) * 100
        for i in range(dim):
            query[i + dim] += query[i] + 1

        # As batch (1D array becomes batch of 1)
        batch_result = tree.batch_query(query)
        assert len(batch_result) == 1

        # As single query
        single_result = tree.query(query)
        assert set(batch_result[0]) == set(single_result)


class TestEdgeCaseBatchQuery:
    """Test batch query with edge cases."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_on_empty_tree(self, PRTree, dim):
        """空のツリーへのバッチクエリが空のリストを返すことを確認."""
        tree = PRTree()

        queries = np.random.rand(5, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        results = tree.batch_query(queries)
        assert len(results) == 5
        for result in results:
            assert result == []

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_large_batch(self, PRTree, dim):
        """大量のクエリがバッチ処理できることを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Large batch
        n_queries = 1000
        queries = np.random.rand(n_queries, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        results = tree.batch_query(queries)
        assert len(results) == n_queries
