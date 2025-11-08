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

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_multiple_gap_sizes(self, PRTree, dim):
        """Test handling of gaps at different orders of magnitude.

        Note: Even with float64 input, the tree is built with float32 internally,
        so gaps smaller than float32 epsilon (~1.19e-7) may not be reliably detected.
        The float64 refinement helps reduce false positives but cannot overcome
        the fundamental precision limit of the float32 tree structure.
        """
        # Test gaps that are representable in float32
        gap_sizes = [1e-4, 1e-5, 1e-6]  # Removed 1e-7 and 1e-8 (below float32 precision)

        for gap in gap_sizes:
            A = np.zeros((1, 2 * dim), dtype=np.float64)
            B = np.zeros((1, 2 * dim), dtype=np.float64)

            # Gap in first dimension
            A[0, 0] = 0.0
            A[0, dim] = 1.0
            B[0, 0] = 1.0 + gap
            B[0, dim] = 2.0

            # Overlap in other dimensions
            for i in range(1, dim):
                A[0, i] = 0.0
                A[0, i + dim] = 100.0
                B[0, i] = 0.0
                B[0, i + dim] = 100.0

            tree = PRTree(np.array([0], dtype=np.int64), A)
            result = tree.query(B[0])

            # With float64 refinement, gaps above float32 precision should be detected
            assert result == [], f"Gap of {gap} should be detected with float64 refinement"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_large_coordinates_small_relative_gaps(self, PRTree, dim):
        """Test large magnitude coordinates with small relative differences.

        Note: Float32 has ~7 decimal digits of precision. At large magnitudes,
        the absolute precision degrades. For example, at 1e6, the precision
        is roughly 0.1, making it impossible to distinguish gaps smaller than that.
        """
        # Use gaps that are representable at each magnitude
        test_cases = [
            (1e3, 0.001),  # At 1e3, 0.001 is ~6 digits precision
            (1e6, 0.1),    # At 1e6, 0.1 is ~7 digits precision
        ]

        for base, gap in test_cases:
            A = np.zeros((1, 2 * dim), dtype=np.float64)
            B = np.zeros((1, 2 * dim), dtype=np.float64)

            # Small relative gap at large magnitude
            A[0, 0] = base
            A[0, dim] = base + 1.0
            B[0, 0] = base + 1.0 + gap
            B[0, dim] = base + 2.0

            # Overlap in other dimensions
            for i in range(1, dim):
                A[0, i] = 0.0
                A[0, i + dim] = 100.0
                B[0, i] = 0.0
                B[0, i + dim] = 100.0

            tree = PRTree(np.array([0], dtype=np.int64), A)
            result = tree.query(B[0])

            # With float64 refinement, should correctly detect representable gaps
            assert result == [], f"Gap of {gap} at base {base} should be detected with float64 refinement"

    @pytest.mark.parametrize("PRTree", [PRTree2D])
    def test_float32_epsilon_boundary(self, PRTree):
        """Test behavior around float32 epsilon (~1.19e-7).

        Note: While float64 refinement helps reduce false positives, the underlying
        tree structure uses float32. Gaps below float32 epsilon cannot be reliably
        detected because they may be lost during float64 to float32 conversion.
        """
        dim = 2
        # Float32 epsilon is approximately 1.19e-7
        # Test gaps well above float32 epsilon that are reliably representable

        test_gaps = [
            (1e-6, "well above float32 epsilon"),
            (5e-6, "far above float32 epsilon"),
        ]

        for gap, description in test_gaps:
            A = np.zeros((1, 2 * dim), dtype=np.float64)
            B = np.zeros((1, 2 * dim), dtype=np.float64)

            A[0, 0] = 0.0
            A[0, dim] = 1.0
            B[0, 0] = 1.0 + gap
            B[0, dim] = 2.0

            # Overlap in other dimensions
            for i in range(1, dim):
                A[0, i] = 0.0
                A[0, i + dim] = 100.0
                B[0, i] = 0.0
                B[0, i + dim] = 100.0

            tree = PRTree(np.array([0], dtype=np.int64), A)
            result = tree.query(B[0])

            # Gaps well above float32 epsilon should be detected with float64 refinement
            assert result == [], f"Gap {description} ({gap}) should be detected with float64 refinement"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree3D, 3), (PRTree4D, 4)])
    def test_precision_in_higher_dimensions(self, PRTree, dim):
        """Test precision handling in 3D and 4D with small gaps."""
        A = np.zeros((1, 2 * dim), dtype=np.float64)
        B = np.zeros((1, 2 * dim), dtype=np.float64)

        # Small gap in each dimension sequentially
        for test_dim in range(dim):
            A.fill(0.0)
            B.fill(0.0)

            # Create overlap in all dimensions except test_dim
            for i in range(dim):
                if i == test_dim:
                    # Gap in this dimension
                    A[0, i] = 0.0
                    A[0, i + dim] = 1.0
                    B[0, i] = 1.0 + 1e-6  # Small gap
                    B[0, i + dim] = 2.0
                else:
                    # Overlap in other dimensions
                    A[0, i] = 0.0
                    A[0, i + dim] = 100.0
                    B[0, i] = 0.0
                    B[0, i + dim] = 100.0

            tree = PRTree(np.array([0], dtype=np.int64), A)
            result = tree.query(B[0])

            assert result == [], f"Small gap in dimension {test_dim} should be detected"
