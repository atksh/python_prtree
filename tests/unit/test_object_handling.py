"""Unit tests for PRTree object handling (set_obj/get_obj)."""
import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D


class TestNormalObjectHandling:
    """Test normal object handling scenarios."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_insert_with_object(self, PRTree, dim):
        """Verify that insert with object works."""
        tree = PRTree()

        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        obj = {"name": "test", "value": 123}
        tree.insert(bb=box, obj=obj)

        assert tree.size() == 1

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_query_with_return_obj(self, PRTree, dim):
        """Verify that object with return_obj=Trueis returned."""
        tree = PRTree()

        boxes_and_objs = [
            (np.array([0.0] * dim + [1.0] * dim), {"id": 1, "name": "obj1"}),
            (np.array([2.0] * dim + [3.0] * dim), {"id": 2, "name": "obj2"}),
        ]

        for box, obj in boxes_and_objs:
            tree.insert(bb=box, obj=obj)

        # Query that intersects first box
        query_box = np.array([0.5] * dim + [0.6] * dim)
        results = tree.query(query_box, return_obj=True)

        assert len(results) == 1
        assert results[0] == {"id": 1, "name": "obj1"}

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_set_and_get_obj(self, PRTree, dim):
        """Verify that set_obj and get_objworks."""
        n = 5
        idx = np.arange(n)
        boxes = np.random.rand(n, 2 * dim) * 100
        for i in range(dim):
            boxes[:, i + dim] += boxes[:, i] + 1

        tree = PRTree(idx, boxes)

        # Set objects
        objs = [{"id": i, "data": f"item_{i}"} for i in range(n)]
        for i, obj in enumerate(objs):
            tree.set_obj(i, obj)

        # Get objects
        for i, expected_obj in enumerate(objs):
            retrieved_obj = tree.get_obj(i)
            assert retrieved_obj == expected_obj


class TestObjectTypes:
    """Test various object types."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_dict_object(self, PRTree, dim):
        """Verify that dict object can be stored and retrieved."""
        tree = PRTree()
        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        obj = {"key": "value", "number": 42}
        tree.insert(bb=box, obj=obj)

        result = tree.query(box, return_obj=True)
        assert result[0] == obj

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_tuple_object(self, PRTree, dim):
        """Verify that tuple object can be stored and retrieved."""
        tree = PRTree()
        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        obj = (1, 2, "three")
        tree.insert(bb=box, obj=obj)

        result = tree.query(box, return_obj=True)
        assert result[0] == obj

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_list_object(self, PRTree, dim):
        """Verify that list object can be stored and retrieved."""
        tree = PRTree()
        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        obj = [1, 2, 3, "four"]
        tree.insert(bb=box, obj=obj)

        result = tree.query(box, return_obj=True)
        assert result[0] == obj

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_nested_object(self, PRTree, dim):
        """Verify that nested object can be stored and retrieved."""
        tree = PRTree()
        box = np.zeros(2 * dim)
        for i in range(dim):
            box[i] = 0.0
            box[i + dim] = 1.0

        obj = {"nested": {"deep": {"value": 123}}, "list": [1, 2, 3]}
        tree.insert(bb=box, obj=obj)

        result = tree.query(box, return_obj=True)
        assert result[0] == obj


class TestObjectPersistence:
    """Test object persistence through save/load."""

    @pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
    def test_objects_not_persisted_in_file(self, PRTree, dim, tmp_path):
        """Verify that objects are not persisted in file (by design)."""
        tree = PRTree()

        boxes_and_objs = [
            (np.array([0.0] * dim + [1.0] * dim), {"id": 1}),
            (np.array([2.0] * dim + [3.0] * dim), {"id": 2}),
        ]

        for box, obj in boxes_and_objs:
            tree.insert(bb=box, obj=obj)

        # Save and load
        fname = tmp_path / "tree.bin"
        tree.save(str(fname))
        loaded_tree = PRTree(str(fname))

        # Objects should not be persisted
        query_box = np.array([0.5] * dim + [0.6] * dim)

        # Query without return_obj should work
        result_idx = loaded_tree.query(query_box)
        assert len(result_idx) > 0

        # Query with return_obj will return (idx, None) tuples
        result_obj = loaded_tree.query(query_box, return_obj=True)
        # Objects were not saved, so they should be None or (idx, None)
        for item in result_obj:
            if isinstance(item, tuple):
                assert item[1] is None or item[1] == (item[0], None)
