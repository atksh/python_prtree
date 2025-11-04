"""Real-world user scenario tests to prevent bugs in actual usage.

These tests simulate how users actually use the library to ensure
they don't encounter unexpected behavior or bugs.
"""
import numpy as np
import pytest
import tempfile
import os

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestQuickStartScenarios:
    """Test scenarios from README Quick Start section."""

    def test_readme_basic_example_works(self):
        """Verify that README basic example works correctly."""
        # Exact code from README
        import numpy as np
        from python_prtree import PRTree2D

        # Create rectangles: [xmin, ymin, xmax, ymax]
        rects = np.array([
            [0.0, 0.0, 1.0, 0.5],  # Rectangle 1
            [1.0, 1.5, 1.2, 3.0],  # Rectangle 2
        ])
        indices = np.array([1, 2])

        # Build the tree
        tree = PRTree2D(indices, rects)

        # Query: find rectangles overlapping with [0.5, 0.2, 0.6, 0.3]
        result = tree.query([0.5, 0.2, 0.6, 0.3])
        assert result == [1], f"Expected [1], got {result}"

        # Batch query (faster for multiple queries)
        queries = np.array([
            [0.5, 0.2, 0.6, 0.3],
            [0.8, 0.5, 1.5, 3.5],
        ])
        results = tree.batch_query(queries)
        assert results == [[1], [1, 2]], f"Expected [[1], [1, 2]], got {results}"

    def test_readme_point_query_example(self):
        """Verify that README point query example works."""
        rects = np.array([[0.0, 0.0, 1.0, 0.5], [1.0, 1.5, 1.2, 3.0]])
        tree = PRTree2D(np.array([1, 2]), rects)

        # Query with point coordinates
        result = tree.query([0.5, 0.5])
        assert isinstance(result, list)

        # Varargs also supported (2D only)
        result2 = tree.query(0.5, 0.5)
        assert isinstance(result2, list)

    def test_readme_dynamic_updates_example(self):
        """Verify that README dynamic update example works."""
        rects = np.array([[0.0, 0.0, 1.0, 0.5], [1.0, 1.5, 1.2, 3.0]])
        tree = PRTree2D(np.array([1, 2]), rects)

        # Insert new rectangle
        tree.insert(3, np.array([1.0, 1.0, 2.0, 2.0]))
        assert tree.size() == 3

        # Remove rectangle by index
        tree.erase(2)
        assert tree.size() == 2

        # Rebuild for optimal performance after many updates
        tree.rebuild()
        assert tree.size() == 2

    def test_readme_store_objects_example(self):
        """Verify that README object storage example works."""
        # Store any picklable Python object with rectangles
        tree = PRTree2D()
        tree.insert(bb=[0, 0, 1, 1], obj={"name": "Building A", "height": 100})
        tree.insert(bb=[2, 2, 3, 3], obj={"name": "Building B", "height": 200})

        # Query and retrieve objects
        results = tree.query([0.5, 0.5, 2.5, 2.5], return_obj=True)
        assert len(results) == 2
        assert {"name": "Building A", "height": 100} in results
        assert {"name": "Building B", "height": 200} in results

    def test_readme_intersections_example(self):
        """Verify that README intersection detection example works."""
        rects = np.array([
            [0.0, 0.0, 2.0, 2.0],  # Large box overlapping others
            [1.0, 1.0, 3.0, 3.0],  # Overlaps with box 1
            [5.0, 5.0, 6.0, 6.0],  # Separate box
        ])
        tree = PRTree2D(np.array([1, 2, 3]), rects)

        # Find all pairs of intersecting rectangles
        pairs = tree.query_intersections()
        assert pairs.shape[1] == 2
        # Should find intersection between boxes 1 and 2
        assert len(pairs) >= 1

    def test_readme_save_load_example(self):
        """Verify that README save/load example works."""
        rects = np.array([[0.0, 0.0, 1.0, 0.5], [1.0, 1.5, 1.2, 3.0]])
        tree = PRTree2D(np.array([1, 2]), rects)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'spatial_index.bin')

            # Save tree to file
            tree.save(filepath)

            # Load from file
            tree_loaded = PRTree2D(filepath)
            assert tree_loaded.size() == 2

            # Or load later
            tree2 = PRTree2D()
            tree2.load(filepath)
            assert tree2.size() == 2


class TestCommonUserMistakes:
    """Test common mistakes users might make."""

    def test_inverted_coordinates_raises_error(self):
        """Verify that wrong coordinates (min > max)raises an error."""
        tree = PRTree2D()

        # Wrong - will raise error
        with pytest.raises((ValueError, RuntimeError)):
            tree.insert(1, [1, 1, 0, 0])  # xmin > xmax, ymin > ymax

    def test_query_before_insert_returns_empty(self):
        """Verify that query before insert returns empty."""
        tree = PRTree2D()
        result = tree.query([0, 0, 1, 1])
        assert result == []

    def test_query_nonexistent_region_returns_empty(self):
        """Verify that query in non-existent region returns empty."""
        tree = PRTree2D(np.array([1]), np.array([[0, 0, 1, 1]]))
        result = tree.query([10, 10, 11, 11])  # Far away
        assert result == []

    def test_erase_nonexistent_index_handled(self):
        """Verify that erase of non-existent index is handled appropriately."""
        tree = PRTree2D(np.array([1, 2]), np.array([[0, 0, 1, 1], [2, 2, 3, 3]]))

        # Try to erase non-existent index
        try:
            tree.erase(999)
            # If it doesn't raise, that's okay (might be no-op)
        except (ValueError, RuntimeError, KeyError):
            # If it raises, that's also okay (explicit error)
            pass

    def test_empty_batch_query_works(self):
        """Verify that empty batch query works."""
        tree = PRTree2D(np.array([1]), np.array([[0, 0, 1, 1]]))

        # Empty query array
        queries = np.empty((0, 4))
        results = tree.batch_query(queries)
        assert len(results) == 0


class TestRealWorldWorkflows:
    """Test realistic workflows users might perform."""

    def test_gis_building_footprints_workflow(self):
        """Test GIS building footprints workflow.."""
        # Simulate GIS data: building footprints
        buildings = [
            {"id": 1, "name": "City Hall", "bounds": [100, 100, 150, 150]},
            {"id": 2, "name": "Library", "bounds": [200, 200, 250, 240]},
            {"id": 3, "name": "Park", "bounds": [120, 120, 180, 180]},
            {"id": 4, "name": "School", "bounds": [300, 300, 350, 350]},
        ]

        # Index buildings
        tree = PRTree2D()
        for building in buildings:
            tree.insert(
                idx=building["id"],
                bb=building["bounds"],
                obj=building
            )

        # User clicks on map at (130, 130)
        click_area = [125, 125, 135, 135]
        results = tree.query(click_area, return_obj=True)

        # Should find City Hall and Park
        found_names = [b["name"] for b in results]
        assert "City Hall" in found_names
        assert "Park" in found_names
        assert "Library" not in found_names

    def test_collision_detection_game_workflow(self):
        """Test game collision detection workflow.."""
        # Game entities with bounding boxes
        tree = PRTree2D()
        tree.insert(1, [10, 10, 20, 20], obj="Player")
        tree.insert(2, [30, 30, 40, 40], obj="Enemy")
        tree.insert(3, [15, 15, 25, 25], obj="PowerUp")

        # Check what player collides with
        player_box = [10, 10, 20, 20]
        collisions = tree.query(player_box, return_obj=True)

        assert "Player" in collisions
        assert "PowerUp" in collisions
        assert "Enemy" not in collisions

    def test_dynamic_scene_with_moving_objects(self):
        """Test dynamic scene with moving objects.."""
        tree = PRTree2D()

        # Initial positions
        tree.insert(1, [0, 0, 10, 10], obj="Object1")
        tree.insert(2, [20, 20, 30, 30], obj="Object2")

        # Object 1 moves - remove old, insert new
        tree.erase(1)
        tree.insert(1, [5, 5, 15, 15], obj="Object1_moved")

        # Query new position
        result = tree.query([10, 10, 12, 12], return_obj=True)
        assert "Object1_moved" in result

    def test_incremental_data_loading(self):
        """Test incremental data loading.."""
        tree = PRTree2D()

        # Load data in batches
        for batch_id in range(5):
            for i in range(10):
                idx = batch_id * 10 + i
                x = i * 10.0
                tree.insert(idx, [x, x, x + 5, x + 5])

        assert tree.size() == 50

        # Query works correctly
        result = tree.query([15, 15, 20, 20])
        assert len(result) > 0

    def test_save_reload_continue_workflow(self):
        """Test save→load→continue workflow.."""
        # Create and populate tree
        tree = PRTree2D()
        for i in range(10):
            tree.insert(i, [i, i, i + 1, i + 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'tree.bin')

            # Save
            tree.save(filepath)

            # Load in new session
            tree2 = PRTree2D(filepath)
            assert tree2.size() == 10

            # Continue adding data
            tree2.insert(10, [10, 10, 11, 11])
            assert tree2.size() == 11

            # Query works
            result = tree2.query([5, 5, 6, 6])
            assert 5 in result


class TestEdgeCases:
    """Test edge cases that users might encounter."""

    def test_touching_boxes_behavior(self):
        """Test touching boxes behavior.."""
        tree = PRTree2D()
        tree.insert(1, [0, 0, 1, 1])
        tree.insert(2, [1, 0, 2, 1])  # Touches box 1 at x=1

        # Query at the touching edge
        result = tree.query([0.5, 0.5, 1.5, 0.5])
        # Both boxes should be found (closed interval semantics)
        assert 1 in result
        assert 2 in result

    def test_very_small_boxes(self):
        """Test very small boxes.."""
        tree = PRTree2D()
        tree.insert(1, [0.0, 0.0, 0.001, 0.001])
        tree.insert(2, [0.01, 0.01, 0.011, 0.011])

        result = tree.query([0.0, 0.0, 0.001, 0.001])
        assert 1 in result
        assert 2 not in result

    def test_very_large_coordinates(self):
        """Test very large coordinates.."""
        tree = PRTree2D()
        large_val = 1e6
        tree.insert(1, [large_val, large_val, large_val + 100, large_val + 100])

        result = tree.query([large_val + 50, large_val + 50, large_val + 60, large_val + 60])
        assert 1 in result

    def test_many_overlapping_boxes(self):
        """Test many overlapping boxes.."""
        tree = PRTree2D()

        # 100 boxes all overlapping at origin
        for i in range(100):
            tree.insert(i, [-1, -1, 1, 1])

        # Query should find all of them
        result = tree.query([0, 0, 0.5, 0.5])
        assert len(result) == 100

    def test_sparse_distribution(self):
        """Test sparse distribution.."""
        tree = PRTree2D()

        # Boxes far apart
        positions = [0, 1000, 2000, 3000, 4000]
        for i, pos in enumerate(positions):
            tree.insert(i, [pos, pos, pos + 1, pos + 1])

        # Query specific regions
        result = tree.query([2000, 2000, 2001, 2001])
        assert result == [2]

    def test_empty_to_full_to_empty_cycle(self):
        """Test empty→full→empty cycle.."""
        tree = PRTree2D()

        # Start empty
        assert tree.size() == 0

        # Fill with data
        for i in range(50):
            tree.insert(i, [i, i, i + 1, i + 1])
        assert tree.size() == 50

        # Empty by erasing all
        for i in range(50):
            tree.erase(i)
        assert tree.size() == 0

        # Can still query
        result = tree.query([0, 0, 1, 1])
        assert result == []

        # Can add again
        tree.insert(100, [0, 0, 1, 1])
        assert tree.size() == 1


class Test3DAnd4DScenarios:
    """Test 3D and 4D specific scenarios."""

    def test_3d_voxel_grid(self):
        """Test 3D voxel grid.."""
        tree = PRTree3D()

        # Create 3D voxel grid
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    idx = x * 25 + y * 5 + z
                    tree.insert(idx, [x, y, z, x + 1, y + 1, z + 1])

        assert tree.size() == 125

        # Query a region
        result = tree.query([2, 2, 2, 3, 3, 3])
        assert len(result) > 0

    def test_4d_spacetime(self):
        """Test 4D spacetime data.."""
        tree = PRTree4D()

        # Objects with position (x, y, z) and time (t)
        tree.insert(1, [0, 0, 0, 0, 1, 1, 1, 10])  # Position at time 0-10
        tree.insert(2, [2, 2, 2, 5, 3, 3, 3, 15])  # Position at time 5-15

        # Query at specific time and space
        result = tree.query([0.5, 0.5, 0.5, 5, 0.6, 0.6, 0.6, 6])
        assert 1 in result
        assert 2 not in result


def test_all_readme_examples_work():
    """Verify that all examples in README work."""
    # This is a meta-test that ensures all README examples are tested
    # We've covered them in TestQuickStartScenarios
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
