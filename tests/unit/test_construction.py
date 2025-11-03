"""Unit tests for PRTree construction/initialization.

Tests cover:
- Normal construction with valid inputs
- Error cases with invalid inputs
- Boundary cases (empty, single element, large datasets)
- Precision cases (float32 vs float64)
- Edge cases (degenerate boxes, identical positions)
"""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestNormalConstruction:
    """Test normal construction scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_valid_inputs(self, PRTree, dim):
        """正常な入力でツリーが構築できることを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n
        assert len(tree) == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_empty_construction(self, PRTree, dim):
        """空のツリーが構築できることを確認."""
        tree = PRTree()
        assert tree.size() == 0
        assert len(tree) == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_single_element_construction(self, PRTree, dim):
        """1要素でツリーが構築できることを確認."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = 0.0
            boxes[0, i + dim] = 1.0

        tree = PRTree(idx, boxes)
        assert tree.size() == 1
        assert len(tree) == 1


class TestErrorConstruction:
    """Test construction with invalid inputs."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_nan_coordinates(self, PRTree, dim):
        """NaN座標での構築がエラーになることを確認."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        boxes[0, 0] = np.nan

        with pytest.raises((ValueError, RuntimeError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_inf_coordinates(self, PRTree, dim):
        """Inf座標での構築がエラーになることを確認."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        boxes[0, 0] = np.inf

        with pytest.raises((ValueError, RuntimeError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_inverted_box(self, PRTree, dim):
        """min > maxのボックスでの構築がエラーになることを確認."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = 10.0  # min
            boxes[0, i + dim] = 0.0  # max (invalid: min > max)

        with pytest.raises((ValueError, RuntimeError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_mismatched_dimensions(self, PRTree, dim):
        """次元数が合わない入力でエラーになることを確認."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, dim))  # Wrong dimension (should be 2*dim)

        with pytest.raises((ValueError, RuntimeError, IndexError)):
            PRTree(idx, boxes)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_mismatched_lengths(self, PRTree, dim):
        """インデックスとボックスの長さが異なる場合にエラーになることを確認."""
        idx = np.array([1, 2, 3])
        boxes = np.zeros((2, 2 * dim))  # Mismatched length

        with pytest.raises((ValueError, RuntimeError, IndexError)):
            PRTree(idx, boxes)


class TestBoundaryConstruction:
    """Test construction with boundary values."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_large_dataset(self, PRTree, dim):
        """大量の要素でツリーが構築できることを確認."""
        n = 10000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_very_small_coordinates(self, PRTree, dim):
        """非常に小さい座標値でツリーが構築できることを確認."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = -1e10
            boxes[0, i + dim] = -1e10 + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == 1

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_very_large_coordinates(self, PRTree, dim):
        """非常に大きい座標値でツリーが構築できることを確認."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = 1e10
            boxes[0, i + dim] = 1e10 + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == 1


class TestPrecisionConstruction:
    """Test construction with different precision."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_float32(self, PRTree, dim):
        """float32でツリーが構築できることを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float32) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_float64(self, PRTree, dim):
        """float64でツリーが構築できることを確認."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim).astype(np.float64) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_int_indices(self, PRTree, dim):
        """整数型のインデックスでツリーが構築できることを確認."""
        n = 10
        idx = np.arange(n, dtype=np.int32)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n


class TestEdgeCaseConstruction:
    """Test construction with edge cases."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_degenerate_boxes(self, PRTree, dim):
        """退化したボックス（min==max）でツリーが構築できることを確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100

        # Make all boxes degenerate (zero volume)
        for i in range(dim):
            boxes[:, i + dim] = boxes[:, i]

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_identical_boxes(self, PRTree, dim):
        """すべて同じボックスでツリーが構築できることを確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.zeros((n, 2 * dim))

        # All boxes are identical
        for i in range(dim):
            boxes[:, i] = 0.0
            boxes[:, i + dim] = 1.0

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_overlapping_boxes(self, PRTree, dim):
        """重なり合うボックスでツリーが構築できることを確認."""
        n = 10
        idx = np.arange(n)
        boxes = np.zeros((n, 2 * dim))

        # All boxes overlap at origin
        for i in range(n):
            for d in range(dim):
                boxes[i, d] = -1.0 - i * 0.1
                boxes[i, d + dim] = 1.0 + i * 0.1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_negative_indices(self, PRTree, dim):
        """負のインデックスでツリーが構築できることを確認."""
        n = 10
        idx = np.arange(-n, 0)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_construction_with_duplicate_indices(self, PRTree, dim):
        """重複したインデックスでの構築（動作は実装依存）."""
        n = 5
        idx = np.array([1, 1, 2, 2, 3])  # Duplicate indices
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        # This may or may not raise an error depending on implementation
        # Just ensure it doesn't crash
        try:
            tree = PRTree(idx, boxes)
            # If it succeeds, size should match input
            assert tree.size() > 0
        except (ValueError, RuntimeError):
            # If it fails, that's also acceptable behavior
            pass
