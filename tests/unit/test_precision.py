"""Unit tests for PRTree precision handling (float32 vs float64)."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestFloat32Precision:
    """Test float32 precision handling."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_float32(self, PRTree, dim):
        """Verify that tree can be constructed with float32."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float32) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_with_float32(self, PRTree, dim):
        """Verify that query with float32works."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float32) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        query_box = np.random.rand(2 * dim).astype(np.float32) * 100
        for i in range(dim):
            query_box[i + dim] += query_box[i] + 1

        result = tree.query(query_box)
        assert isinstance(result, list)


class TestFloat64Precision:
    """Test float64 precision handling."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_float64(self, PRTree, dim):
        """Verify that tree can be constructed with float64."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float64) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_small_gap_with_float64(self, PRTree, dim):
        """Verify that small gap with float64 is handled correctly."""
        A = np.zeros((1, 2 * dim), dtype=np.float64)
        B = np.zeros((1, 2 * dim), dtype=np.float64)

        # Small gap in first dimension (< 1e-5)
        A[0, 0] = 0.0
        A[0, dim] = 75.02750896
        B[0, 0] = 75.02751435
        B[0, dim] = 100.0

        # Fill other dimensions to ensure overlap
        for i in range(1, dim):
            A[0, i] = 0.0
            A[0, i + dim] = 100.0
            B[0, i] = 0.0
            B[0, i + dim] = 100.0

        tree = PRTree(np.array([0]), A)
        result = tree.query(B[0])

        # Should not intersect due to small gap
        assert result == []

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_large_magnitude_coordinates_float64(self, PRTree, dim):
        """Verify that large magnitude coordinates with float64 are handled correctly."""
        A = np.zeros((1, 2 * dim), dtype=np.float64)
        B = np.zeros((1, 2 * dim), dtype=np.float64)

        base = 1e6
        for i in range(dim):
            A[0, i] = base + i
            A[0, i + dim] = base + i + 1.0
            B[0, i] = base + i + 1.1
            B[0, i + dim] = base + i + 2.0

        tree = PRTree(np.array([0]), A)
        result = tree.query(B[0])

        # Should not intersect
        assert result == []


class TestMixedPrecision:
    """Test mixed precision scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_float32_tree_float64_query(self, PRTree, dim):
        """Verify that float64 query on float32 treeworks."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float32) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Query with float64
        query_box = np.random.rand(2 * dim).astype(np.float64) * 100
        for i in range(dim):
            query_box[i + dim] += query_box[i] + 1

        result = tree.query(query_box)
        assert isinstance(result, list)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_float64_tree_float32_query(self, PRTree, dim):
        """Verify that float32 query on float64 treeworks."""
        n = 50
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float64) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Query with float32
        query_box = np.random.rand(2 * dim).astype(np.float32) * 100
        for i in range(dim):
            query_box[i + dim] += query_box[i] + 1

        result = tree.query(query_box)
        assert isinstance(result, list)


class TestPrecisionEdgeCases:
    """Test precision edge cases."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_degenerate_boxes_float64(self, PRTree, dim):
        """Verify that degenerate boxes with float64 are handled correctly."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float64) * 100

        # Make degenerate (min == max)
        for i in range(dim):
            boxes[:, i + dim] = boxes[:, i]

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_touching_boxes_float64(self, PRTree, dim):
        """Verify that touching boxes with float64 are handled correctly."""
        A = np.zeros((1, 2 * dim), dtype=np.float64)
        B = np.zeros((1, 2 * dim), dtype=np.float64)

        for i in range(dim):
            A[0, i] = 0.0
            A[0, i + dim] = 1.0
            B[0, i] = 1.0
            B[0, i + dim] = 2.0

        tree = PRTree(np.array([0]), A)
        result = tree.query(B[0])

        # Should intersect (closed interval semantics)
        assert result == [0]
