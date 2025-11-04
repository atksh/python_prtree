"""Comprehensive memory safety tests discovered after finding segfaults.

These tests ensure complete memory safety across all operations and edge cases.
After discovering 2 critical segfaults, this file adds exhaustive safety testing.
"""
import numpy as np
import pytest
import gc

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestEmptyTreeOperations:
    """Test ALL operations on empty trees to prevent segfaults."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_all_query_operations_on_empty_tree(self, PRTree, dim):
        """Verify that all query operations work safely on empty tree."""
        tree = PRTree()

        # Single query with box
        query_box = np.zeros(2 * dim)
        for i in range(dim):
            query_box[i] = 0.0
            query_box[i + dim] = 1.0

        result = tree.query(query_box)
        assert result == []

        # Point query (2D only for varargs)
        if dim == 2:
            result = tree.query(0.5, 0.5)
            assert result == []

        # Query with tuple
        result = tree.query(tuple(query_box))
        assert result == []

        # Query with list
        result = tree.query(list(query_box))
        assert result == []

        # Query with return_obj
        result = tree.query(query_box, return_obj=True)
        assert result == []

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_variations_on_empty_tree(self, PRTree, dim):
        """Verify that all batch query variations work safely on empty tree."""
        tree = PRTree()

        # Batch query with multiple queries
        queries = np.random.rand(10, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        results = tree.batch_query(queries)
        assert len(results) == 10
        assert all(r == [] for r in results)

        # Batch query with single query
        single_query = queries[0:1]
        results = tree.batch_query(single_query)
        assert len(results) == 1
        assert results[0] == []

        # Batch query with empty array
        empty_queries = np.empty((0, 2 * dim))
        results = tree.batch_query(empty_queries)
        assert len(results) == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_intersections_on_empty_tree(self, PRTree, dim):
        """Verify that query_intersections works safely on empty tree."""
        tree = PRTree()
        pairs = tree.query_intersections()
        assert pairs.shape == (0, 2)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_properties_on_empty_tree(self, PRTree, dim):
        """Verify that properties work safely on empty tree."""
        tree = PRTree()
        assert tree.size() == 0
        assert len(tree) == 0
        assert tree.n == 0

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_erase_on_empty_tree(self, PRTree, dim):
        """Verify that erase from empty treeproperly returns error."""
        tree = PRTree()
        with pytest.raises(ValueError):
            tree.erase(1)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_rebuild_on_empty_tree(self, PRTree, dim):
        """Verify that rebuild on empty treeworks safely."""
        tree = PRTree()
        try:
            tree.rebuild()
            # If it doesn't crash, that's good
        except (RuntimeError, ValueError):
            # Expected for empty trees
            pass


class TestSingleElementTreeOperations:
    """Test operations on single-element trees (another critical edge case)."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_all_operations_on_single_element_tree(self, PRTree, dim):
        """Verify that all operations on single-element treeworks safely."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        tree.insert(idx=1, bb=box)

        # Query operations
        result = tree.query(box)
        assert 1 in result

        # Batch query
        queries = np.array([box, box])
        results = tree.batch_query(queries)
        assert len(results) == 2
        assert all(1 in r for r in results)

        # Query intersections (no self-intersections)
        pairs = tree.query_intersections()
        assert pairs.shape[0] == 0

        # Properties
        assert tree.size() == 1
        assert len(tree) == 1

        # Rebuild
        tree.rebuild()
        assert tree.size() == 1

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_can_erase_last_element(self, PRTree, dim):
        """Test ability to erase last element (limitation fixed!)."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        tree.insert(idx=1, bb=box)
        assert tree.size() == 1

        # This now works! Limitation fixed.
        tree.erase(1)
        assert tree.size() == 0

        # Verify tree is truly empty
        result = tree.query(box)
        assert result == []


class TestBoundaryValues:
    """Test with extreme boundary values to ensure no overflow/underflow."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_very_large_coordinates(self, PRTree, dim):
        """Verify safety with very large coordinates."""
        large_val = 1e10

        idx = np.array([1])
        boxes = np.full((1, 2 * dim), large_val)
        for i in range(dim):
            boxes[0, i] = large_val
            boxes[0, i + dim] = large_val + 100

        tree = PRTree(idx, boxes)
        result = tree.query(boxes[0])
        assert 1 in result

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_very_small_coordinates(self, PRTree, dim):
        """Verify safety with very small coordinates."""
        small_val = 1e-10

        idx = np.array([1])
        boxes = np.full((1, 2 * dim), small_val)
        for i in range(dim):
            boxes[0, i] = small_val
            boxes[0, i + dim] = small_val * 2

        tree = PRTree(idx, boxes)
        result = tree.query(boxes[0])
        assert 1 in result

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_negative_coordinates(self, PRTree, dim):
        """Verify safety with negative coordinates."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = -1000
            boxes[0, i + dim] = -900

        tree = PRTree(idx, boxes)
        result = tree.query(boxes[0])
        assert 1 in result

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_mixed_sign_coordinates(self, PRTree, dim):
        """Verify safety with mixed sign coordinates."""
        idx = np.array([1, 2])
        boxes = np.zeros((2, 2 * dim))
        for i in range(dim):
            boxes[0, i] = -100
            boxes[0, i + dim] = 100
            boxes[1, i] = -50
            boxes[1, i + dim] = 50

        tree = PRTree(idx, boxes)

        # Query that spans negative and positive
        query_box = np.zeros(2 * dim)
        for i in range(dim):
            query_box[i] = -75
            query_box[i + dim] = 75

        result = tree.query(query_box)
        assert 1 in result and 2 in result


class TestMemoryPressure:
    """Test operations under memory pressure."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_rapid_insert_erase_cycles(self, PRTree, dim):
        """Verify memory safety with rapid insert/erase cycles."""
        tree = PRTree()

        # Keep at least 2 elements to avoid erase limitation
        box_keep = np.zeros(2 * dim)
        for i in range(dim):
            box_keep[i] = 1000.0
            box_keep[i + dim] = 1001.0
        tree.insert(idx=9999, bb=box_keep)

        # Rapid insert/erase cycles
        for cycle in range(100):
            # Insert
            box = np.random.rand(2 * dim) * 100
            for i in range(dim):
                box[i + dim] += box[i] + 1
            tree.insert(idx=cycle, bb=box)

            # Query
            result = tree.query(box)
            assert cycle in result

            # Erase
            tree.erase(cycle)

        # Tree should still be valid
        assert tree.size() == 1
        gc.collect()  # Force garbage collection

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2)])
    def test_very_large_batch_query(self, PRTree, dim):
        """Verify safety with very large batch query."""
        n = 1000
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 1000
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Very large batch query
        n_queries = 10000
        queries = np.random.rand(n_queries, 2 * dim) * 1000
        for i in range(dim):
            queries[:, i + dim] += queries[:, i] + 1

        results = tree.batch_query(queries)
        assert len(results) == n_queries


class TestNullAndInvalidInputs:
    """Test handling of null and invalid inputs."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_with_nan(self, PRTree, dim):
        """Verify that query with NaN coordinates works safely or returns error."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = 0.0
            boxes[0, i + dim] = 1.0

        tree = PRTree(idx, boxes)

        # Query with NaN
        query_box = np.full(2 * dim, np.nan)

        try:
            result = tree.query(query_box)
            # If it doesn't crash, result should be empty or raise
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_with_inf(self, PRTree, dim):
        """Verify that query with infinite coordinates works safely or returns error."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = 0.0
            boxes[0, i + dim] = 1.0

        tree = PRTree(idx, boxes)

        # Query with inf
        query_box = np.full(2 * dim, np.inf)

        try:
            result = tree.query(query_box)
            # If it doesn't crash, result should be handled
        except (ValueError, RuntimeError, OverflowError):
            pass  # Expected behavior

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_invalid_dimensions(self, PRTree, dim):
        """Verify that insert with invalid dimensionsproperly returns error."""
        tree = PRTree()

        # Wrong dimension box
        wrong_box = np.zeros(2 * dim + 1)  # One extra dimension

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            tree.insert(idx=1, bb=wrong_box)

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_batch_query_with_wrong_dimensions(self, PRTree, dim):
        """Verify that batch_query with invalid dimensionsproperly returns error."""
        idx = np.array([1])
        boxes = np.zeros((1, 2 * dim))
        for i in range(dim):
            boxes[0, i] = 0.0
            boxes[0, i + dim] = 1.0

        tree = PRTree(idx, boxes)

        # Wrong dimension queries
        wrong_queries = np.zeros((5, 2 * dim + 1))  # One extra dimension

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            tree.batch_query(wrong_queries)


class TestEdgeCaseTransitions:
    """Test transitions between edge cases (empty -> 1 element -> 2 elements)."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_empty_to_one_to_many_elements(self, PRTree, dim):
        """Verify safety during empty → 1 element → many elements transition."""
        tree = PRTree()

        # Empty state - all operations should be safe
        assert tree.size() == 0
        result = tree.query(np.zeros(2 * dim))
        assert result == []
        results = tree.batch_query(np.zeros((5, 2 * dim)))
        assert all(r == [] for r in results)

        # Add first element
        box1 = np.zeros(2 * dim)
        for i in range(dim):
            box1[i] = 0.0
            box1[i + dim] = 1.0
        tree.insert(idx=1, bb=box1)

        # One element state
        assert tree.size() == 1
        result = tree.query(box1)
        assert 1 in result

        # Add second element
        box2 = np.zeros(2 * dim)
        for i in range(dim):
            box2[i] = 2.0
            box2[i + dim] = 3.0
        tree.insert(idx=2, bb=box2)

        # Two elements state
        assert tree.size() == 2
        result1 = tree.query(box1)
        result2 = tree.query(box2)
        assert 1 in result1
        assert 2 in result2

        # Add many more
        for i in range(3, 101):  # 3 to 100 inclusive = 98 more elements + 2 existing = 100 total
            box = np.random.rand(2 * dim) * 100
            for d in range(dim):
                box[d + dim] += box[d] + 1
            tree.insert(idx=i, bb=box)

        assert tree.size() == 100

        # All operations should still work
        queries = np.random.rand(10, 2 * dim) * 100
        for i in range(dim):
            queries[:, i + dim] = np.maximum(queries[:, i + dim], queries[:, i] + 1)
        results = tree.batch_query(queries)
        assert len(results) == 10

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_many_to_few_to_empty_via_erase(self, PRTree, dim):
        """Verify safety during many → few → empty transition."""
        n = 100
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)
        assert tree.size() == n

        # Erase down to 1 element
        for i in range(n - 1):
            tree.erase(i)

        assert tree.size() == 1

        # Can now erase the last element (limitation fixed!)
        tree.erase(n - 1)
        assert tree.size() == 0

        # Verify tree is truly empty
        query_box = np.random.rand(2 * dim) * 100
        result = tree.query(query_box)
        assert result == []


class TestObjectHandlingSafety:
    """Test object storage safety with various object types."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_various_object_types(self, PRTree, dim):
        """Verify safety with various object types."""
        tree = PRTree()

        objects = [
            {"type": "dict"},
            ["list", "with", "items"],
            ("tuple", "with", "items"),
            "simple string",
            42,
            3.14,
            {"nested": {"dict": {"with": "depth"}}},
        ]

        for i, obj in enumerate(objects):
            box = np.zeros(2 * dim)
            for d in range(dim):
                box[d] = i * 10
                box[d + dim] = i * 10 + 5
            tree.insert(idx=i+1, bb=box, obj=obj)  # Always provide idx

        # Query and verify objects
        for i, expected_obj in enumerate(objects):
            box = np.zeros(2 * dim)
            for d in range(dim):
                box[d] = i * 10
                box[d + dim] = i * 10 + 5

            result = tree.query(box, return_obj=True)
            assert len(result) > 0, f"No results for box at index {i}"
            assert expected_obj in result, f"Expected {expected_obj} not found in {result}"


class TestConcurrentOperationsSafety:
    """Test safety under simulated concurrent operations."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2)])
    def test_interleaved_insert_query_operations(self, PRTree, dim):
        """Verify safety with interleaved insert and query operations."""
        tree = PRTree()

        for i in range(100):
            # Insert
            box = np.random.rand(2 * dim) * 100
            for d in range(dim):
                box[d + dim] += box[d] + 1
            tree.insert(idx=i, bb=box)

            # Immediate query
            result = tree.query(box)
            assert i in result

            # Batch query
            queries = np.random.rand(10, 2 * dim) * 100
            for d in range(dim):
                queries[:, d + dim] = np.maximum(queries[:, d + dim], queries[:, d] + 1)
            results = tree.batch_query(queries)
            assert len(results) == 10

            # Query intersections
            pairs = tree.query_intersections()
            assert pairs.shape[1] == 2


# Summary comment
"""
This comprehensive test suite adds extensive memory safety testing after
discovering critical segfaults. Key additions:

1. Empty tree operations (ALL methods)
2. Single-element tree operations
3. Boundary values (large, small, negative, mixed)
4. Memory pressure scenarios
5. Null/invalid inputs
6. Edge case transitions (empty -> 1 -> many -> few -> empty)
7. Object handling safety
8. Concurrent operation patterns

Total new test functions: ~25
Expected test cases (with parametrization): ~75-90 additional tests
"""
