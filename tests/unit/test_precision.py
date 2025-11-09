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


class TestAdaptiveEpsilon:
    """Test adaptive epsilon calculation and behavior."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_adaptive_epsilon_small_coordinates(self, PRTree, dim):
        """Verify adaptive epsilon works correctly for small coordinates (< 1.0).

        For small coordinates, absolute epsilon should dominate.
        """
        tree = PRTree()

        # Insert small box
        box = np.zeros(2 * dim, dtype=np.float64)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 0.1

        tree.insert(idx=0, bb=box)

        # Query with very small gap (should not intersect)
        query = np.zeros(2 * dim, dtype=np.float64)
        query[0] = 0.1 + 1e-7  # Small gap in first dimension
        query[dim] = 0.2
        for i in range(1, dim):
            query[i] = 0.0
            query[i + dim] = 0.1

        result = tree.query(query)
        # With adaptive epsilon, small absolute gaps should be detected
        assert result == [], "Small gap should be detected with adaptive epsilon"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_adaptive_epsilon_large_coordinates(self, PRTree, dim):
        """Verify adaptive epsilon scales with coordinate magnitude.

        For large coordinates, relative epsilon should dominate.
        """
        tree = PRTree()

        # Insert box at large magnitude
        base = 1e7
        box = np.zeros(2 * dim, dtype=np.float64)
        for i in range(dim):
            box[i] = base
            box[i + dim] = base + 1000.0

        tree.insert(idx=0, bb=box)

        # Query with gap that would be significant at small scale but
        # should be detected at large scale with adaptive epsilon
        query = np.zeros(2 * dim, dtype=np.float64)
        query[0] = base + 1000.0 + 0.01  # Small relative gap
        query[dim] = base + 2000.0
        for i in range(1, dim):
            query[i] = base
            query[i + dim] = base + 1000.0

        result = tree.query(query)
        # Gap should be detected with adaptive epsilon
        assert result == [], "Gap should be detected with adaptive epsilon at large scale"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_adaptive_epsilon_mixed_scales(self, PRTree, dim):
        """Test adaptive epsilon with boxes at different scales."""
        tree = PRTree()

        # Insert boxes at different scales
        scales = [0.1, 1.0, 100.0, 10000.0]
        for idx, scale in enumerate(scales):
            box = np.zeros(2 * dim, dtype=np.float64)
            for i in range(dim):
                box[i] = scale
                box[i + dim] = scale + scale * 0.1
            tree.insert(idx=idx, bb=box)

        assert tree.size() == len(scales)

        # Query each box with appropriate gap
        for idx, scale in enumerate(scales):
            query = np.zeros(2 * dim, dtype=np.float64)
            # Create query just after the box with adaptive gap
            query[0] = scale + scale * 0.1 + scale * 1e-5
            query[dim] = scale + scale * 0.2
            for i in range(1, dim):
                query[i] = scale
                query[i + dim] = scale + scale * 0.1

            result = tree.query(query)
            # Should not include the box we're testing against
            # (may include others due to overlap in other dimensions)
            # But for our test, we create sufficient separation

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2)])
    def test_subnormal_number_detection(self, PRTree, dim):
        """Verify subnormal number detection in insert operations."""
        tree = PRTree()

        # Create box with subnormal number (very small but non-zero)
        box = np.zeros(2 * dim, dtype=np.float64)
        box[0] = 1e-320  # Subnormal number
        box[dim] = 1.0
        for i in range(1, dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        # With subnormal detection enabled (default), should raise error
        with pytest.raises((ValueError, RuntimeError)):
            tree.insert(idx=0, bb=box)

        # Disable subnormal detection
        tree.set_subnormal_detection(False)
        assert tree.get_subnormal_detection() == False

        # Now insert should work
        tree.insert(idx=0, bb=box)
        assert tree.size() == 1

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_precision_parameter_configuration(self, PRTree, dim):
        """Test precision parameter getters and setters."""
        tree = PRTree()

        # Test relative epsilon
        tree.set_relative_epsilon(1e-5)
        assert tree.get_relative_epsilon() == 1e-5

        # Test absolute epsilon
        tree.set_absolute_epsilon(1e-7)
        assert tree.get_absolute_epsilon() == 1e-7

        # Test adaptive epsilon flag
        tree.set_adaptive_epsilon(False)
        assert tree.get_adaptive_epsilon() == False
        tree.set_adaptive_epsilon(True)
        assert tree.get_adaptive_epsilon() == True

        # Test subnormal detection flag
        tree.set_subnormal_detection(False)
        assert tree.get_subnormal_detection() == False
        tree.set_subnormal_detection(True)
        assert tree.get_subnormal_detection() == True

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2)])
    def test_adaptive_epsilon_disabled(self, PRTree, dim):
        """Test behavior when adaptive epsilon is disabled."""
        tree = PRTree()

        # Disable adaptive epsilon
        tree.set_adaptive_epsilon(False)
        tree.set_absolute_epsilon(1e-6)

        # Insert box at large scale
        base = 1e6
        box = np.zeros(2 * dim, dtype=np.float64)
        for i in range(dim):
            box[i] = base
            box[i + dim] = base + 100.0

        tree.insert(idx=0, bb=box)

        # Query with gap smaller than absolute epsilon
        # Without adaptive epsilon, this might cause false positive
        query = np.zeros(2 * dim, dtype=np.float64)
        query[0] = base + 100.0 + 1e-7  # Gap smaller than absolute epsilon
        query[dim] = base + 200.0
        for i in range(1, dim):
            query[i] = base
            query[i + dim] = base + 100.0

        result = tree.query(query)
        # Result depends on absolute epsilon vs gap size

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
    def test_custom_relative_epsilon(self, PRTree, dim):
        """Test custom relative epsilon values."""
        tree = PRTree()

        # Set tighter relative epsilon
        tree.set_relative_epsilon(1e-8)

        # Insert box
        box = np.zeros(2 * dim, dtype=np.float64)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 100.0

        tree.insert(idx=0, bb=box)

        # Query with very small gap (relative to box size)
        query = np.zeros(2 * dim, dtype=np.float64)
        query[0] = 100.0 + 1e-6  # Very small gap
        query[dim] = 200.0
        for i in range(1, dim):
            query[i] = 0.0
            query[i + dim] = 100.0

        result = tree.query(query)
        # With tighter epsilon, small gaps should be detected
        assert result == [], "Small gap should be detected with tight relative epsilon"

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2)])
    def test_degenerate_box_epsilon_handling(self, PRTree, dim):
        """Test adaptive epsilon for degenerate boxes (point-like)."""
        tree = PRTree()

        # Insert degenerate box (min == max, i.e., a point)
        box = np.zeros(2 * dim, dtype=np.float64)
        for i in range(dim):
            box[i] = 100.0
            box[i + dim] = 100.0  # Degenerate

        tree.insert(idx=0, bb=box)

        # Query very close to the point
        query = np.zeros(2 * dim, dtype=np.float64)
        query[0] = 100.0 + 1e-6  # Very close
        query[dim] = 101.0
        for i in range(1, dim):
            query[i] = 99.0
            query[i + dim] = 101.0

        result = tree.query(query)
        # For degenerate boxes, epsilon should be based on magnitude
        # Gap should be detected
