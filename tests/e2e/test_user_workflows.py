"""End-to-end tests for common user workflows."""
import gc
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_spatial_indexing_workflow(PRTree, dim):
    """ユーザーワークフロー: 空間インデックスの構築とクエリ."""
    # Simulate a spatial database of objects
    n_objects = 1000
    np.random.seed(42)

    # Create random spatial objects
    idx = np.arange(n_objects)
    boxes = np.random.rand(n_objects, 2 * dim) * 1000
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + np.random.rand(n_objects) * 10

    # Build spatial index
    tree = PRTree(idx, boxes)
    assert tree.size() == n_objects

    # Query for objects in a region
    query_region = np.array([100] * dim + [200] * dim)
    results = tree.query(query_region)

    # Verify all results actually intersect
    def has_intersect(x, y, dim):
        return all([max(x[i], y[i]) <= min(x[i + dim], y[i + dim]) for i in range(dim)])

    for result_idx in results:
        assert has_intersect(boxes[result_idx], query_region, dim)


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_dynamic_updates_workflow(PRTree, dim):
    """ユーザーワークフロー: 動的な更新（挿入・削除）."""
    # Start with empty tree
    tree = PRTree()

    # Simulate adding objects over time
    objects = []
    for i in range(100):
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1

        tree.insert(idx=i, bb=box)
        objects.append(box)

    assert tree.size() == 100

    # Remove some objects
    to_remove = [10, 20, 30, 40, 50]
    for idx in to_remove:
        tree.erase(idx)

    assert tree.size() == 95

    # Add more objects
    for i in range(100, 150):
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1

        tree.insert(idx=i, bb=box)
        objects.append(box)

    assert tree.size() == 145

    # Query should work correctly
    query_box = np.random.rand(2 * dim) * 100
    for d in range(dim):
        query_box[d + dim] += query_box[d] + 1

    results = tree.query(query_box)
    assert isinstance(results, list)

    # Removed indices should not appear
    for idx in to_remove:
        assert idx not in results


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_persistence_workflow(PRTree, dim, tmp_path):
    """ユーザーワークフロー: データの永続化."""
    # Build initial tree
    n = 500
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree = PRTree(idx, boxes)

    # Save to disk
    fname = tmp_path / "spatial_index.bin"
    tree.save(str(fname))

    # Simulate application restart
    del tree
    gc.collect()

    # Load from disk
    loaded_tree = PRTree(str(fname))
    assert loaded_tree.size() == n

    # Use loaded tree
    query_box = np.random.rand(2 * dim) * 100
    for i in range(dim):
        query_box[i + dim] += query_box[i] + 1

    results = loaded_tree.query(query_box)
    assert isinstance(results, list)


def test_collision_detection_workflow_2d():
    """ユーザーワークフロー: 2D衝突検出（ゲーム・シミュレーション）."""
    # Simulate game entities
    entities = {
        "player": [10, 10, 12, 12],
        "enemy1": [50, 50, 52, 52],
        "enemy2": [11, 11, 13, 13],  # Overlaps with player
        "wall1": [0, 0, 1, 100],
        "wall2": [99, 0, 100, 100],
    }

    idx_to_name = {}
    idx = 0
    boxes = []

    for name, box in entities.items():
        idx_to_name[idx] = name
        boxes.append(box)
        idx += 1

    tree = PRTree2D(np.arange(len(boxes)), np.array(boxes))

    # Check collisions with player
    player_box = entities["player"]
    collisions = tree.query(player_box)

    collision_names = [idx_to_name[i] for i in collisions]

    assert "player" in collision_names
    assert "enemy2" in collision_names  # Should collide
    assert "enemy1" not in collision_names  # Should not collide


def test_object_storage_workflow_2d():
    """ユーザーワークフロー: オブジェクト付きの空間インデックス."""
    # Store rich objects with spatial index
    objects = [
        {"id": 1, "type": "building", "name": "City Hall", "box": [0, 0, 10, 10]},
        {"id": 2, "type": "building", "name": "Library", "box": [20, 20, 30, 25]},
        {"id": 3, "type": "park", "name": "Central Park", "box": [5, 5, 15, 15]},
        {"id": 4, "type": "road", "name": "Main Street", "box": [0, 5, 100, 7]},
    ]

    tree = PRTree2D()

    for obj in objects:
        tree.insert(bb=obj["box"], obj=obj)

    # Query for objects in a region
    query_region = [5, 5, 10, 10]
    results = tree.query(query_region, return_obj=True)

    # Extract object data (return_obj=True returns objects directly, not tuples)
    found_objects = results

    # City Hall and Central Park should be found
    found_names = [obj["name"] for obj in found_objects]
    assert "City Hall" in found_names or "Central Park" in found_names


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
def test_batch_processing_workflow(PRTree, dim):
    """ユーザーワークフロー: バッチ処理（大量クエリ）."""
    # Build index
    n = 1000
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 1000
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 10

    tree = PRTree(idx, boxes)

    # Batch query (e.g., processing many search requests)
    n_queries = 5000
    queries = np.random.rand(n_queries, 2 * dim) * 1000
    for i in range(dim):
        queries[:, i + dim] += queries[:, i] + 5

    # Use batch_query for efficiency
    results = tree.batch_query(queries)

    assert len(results) == n_queries
    for result in results:
        assert isinstance(result, list)


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_intersection_detection_workflow(PRTree, dim):
    """ユーザーワークフロー: 全ペアの交差検出."""
    # Simulate checking for overlapping regions
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 5

    tree = PRTree(idx, boxes)

    # Find all intersecting pairs efficiently
    pairs = tree.query_intersections()

    # Process each pair
    for i, j in pairs:
        assert i < j
        # In real application, might resolve conflicts or merge regions


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_rebuild_optimization_workflow(PRTree, dim):
    """ユーザーワークフロー: 多数の更新後の最適化."""
    # Initial index
    n = 500
    idx = np.arange(n)
    boxes = np.random.rand(n, 2 * dim) * 100
    for i in range(dim):
        boxes[:, i + dim] += boxes[:, i] + 1

    tree = PRTree(idx, boxes)

    # Many updates
    for i in range(100):
        tree.erase(i)

    for i in range(100):
        box = np.random.rand(2 * dim) * 100
        for d in range(dim):
            box[d + dim] += box[d] + 1
        tree.insert(idx=n + i, bb=box)

    # Rebuild for better query performance
    tree.rebuild()

    # Verify still works correctly
    query_box = np.random.rand(2 * dim) * 100
    for i in range(dim):
        query_box[i + dim] += query_box[i] + 1

    results = tree.query(query_box)
    assert isinstance(results, list)
