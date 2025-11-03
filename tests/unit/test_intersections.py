"""Unit tests for PRTree query_intersections operations."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


def has_intersect(x, y, dim):
    """Helper function to check if two boxes intersect."""
    return all([max(x[i], y[i]) <= min(x[i + dim], y[i + dim]) for i in range(dim)])


class TestNormalIntersections:
    """Test normal query_intersections scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_returns_correct_pairs(self, PRTree, dim):
        """query_intersectionsが正しいペアを返すことを確認."""
        np.random.seed(42)
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        pairs = tree.query_intersections()

        # Verify output shape
        assert pairs.ndim == 2
        assert pairs.shape[1] == 2

        # Verify all pairs are valid (i < j)
        assert np.all(pairs[:, 0] < pairs[:, 1])

        # Verify correctness
        expected_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if has_intersect(boxes[i], boxes[j], dim):
                    expected_pairs.append((idx[i], idx[j]))

        pairs_set = set(map(tuple, pairs))
        expected_set = set(expected_pairs)
        assert pairs_set == expected_set


class TestBoundaryIntersections:
    """Test query_intersections with boundary cases."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_empty_tree(self, PRTree, dim):
        """空のツリーでquery_intersectionsが空の配列を返すことを確認."""
        tree = PRTree()
        pairs = tree.query_intersections()

        assert pairs.shape == (0, 2)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_no_intersections(self, PRTree, dim):
        """交差しないボックスでquery_intersectionsが空を返すことを確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.zeros((n, 2 * dim))

        # Create well-separated boxes
        for i in range(n):
            for d in range(dim):
                boxes[i, d] = 10 * i + d * 0.1
                boxes[i, d + dim] = 10 * i + d * 0.1 + 1

        tree = PRTree(idx, boxes)
        pairs = tree.query_intersections()

        assert pairs.shape[0] == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_all_intersecting(self, PRTree, dim):
        """すべてのボックスが交差する場合のquery_intersections."""
        n = 10
        idx = np.arange(n)
        boxes = np.zeros((n, 2 * dim))

        # All boxes overlap at origin
        for i in range(n):
            for d in range(dim):
                boxes[i, d] = -1.0 - i * 0.1
                boxes[i, d + dim] = 1.0 + i * 0.1

        tree = PRTree(idx, boxes)
        pairs = tree.query_intersections()

        # All boxes should intersect: n*(n-1)/2 pairs
        expected_count = n * (n - 1) // 2
        assert pairs.shape[0] == expected_count


class TestEdgeCaseIntersections:
    """Test query_intersections with edge cases."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_touching_boxes(self, PRTree, dim):
        """接しているボックスが交差と判定されることを確認."""
        idx = np.array([0, 1])
        boxes = np.zeros((2, 2 * dim))

        # Box 0: [0, 1] in all dimensions
        for d in range(dim):
            boxes[0, d] = 0.0
            boxes[0, d + dim] = 1.0

        # Box 1: [1, 2] in all dimensions (touches box 0)
        for d in range(dim):
            boxes[1, d] = 1.0
            boxes[1, d + dim] = 2.0

        tree = PRTree(idx, boxes)
        pairs = tree.query_intersections()

        # Should be considered intersecting
        assert pairs.shape[0] == 1
        assert tuple(pairs[0]) == (0, 1)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_single_element(self, PRTree, dim):
        """1要素のツリーでquery_intersectionsが空を返すことを確認."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for d in range(dim):
            boxes[0, d] = 0.0
            boxes[0, d + dim] = 1.0

        tree = PRTree(idx, boxes)
        pairs = tree.query_intersections()

        assert pairs.shape[0] == 0


class TestConsistencyIntersections:
    """Test query_intersections consistency."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_after_insert(self, PRTree, dim):
        """挿入後のquery_intersectionsが正しく動作することを確認."""
        np.random.seed(42)
        n = 20
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        pairs_initial = tree.query_intersections()

        # Insert a box that overlaps all
        new_box = np.zeros(2 * dim)
        for d in range(dim):
            new_box[d] = -10.0
            new_box[d + dim] = 110.0

        tree.insert(idx=max(idx) + 1, bb=new_box)

        pairs_after = tree.query_intersections()

        # Should have more pairs now
        assert pairs_after.shape[0] > pairs_initial.shape[0]

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_float64_precision(self, PRTree, dim):
        """float64でquery_intersectionsが正しく動作することを確認."""
        np.random.seed(42)
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float64) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        pairs = tree.query_intersections()

        # Verify correctness
        expected_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if has_intersect(boxes[i], boxes[j], dim):
                    expected_pairs.append((idx[i], idx[j]))

        pairs_set = set(map(tuple, pairs))
        expected_set = set(expected_pairs)
        assert pairs_set == expected_set
