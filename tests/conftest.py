"""Shared pytest fixtures and configuration for all tests."""
import numpy as np
import pytest


@pytest.fixture(params=[(2, "PRTree2D"), (3, "PRTree3D"), (4, "PRTree4D")])
def dimension_and_class(request):
    """Parametrize tests across all dimensions and tree classes."""
    from python_prtree import PRTree2D, PRTree3D, PRTree4D

    dim, class_name = request.param
    tree_classes = {
        "PRTree2D": PRTree2D,
        "PRTree3D": PRTree3D,
        "PRTree4D": PRTree4D,
    }
    return dim, tree_classes[class_name]


@pytest.fixture
def sample_boxes_2d():
    """Generate sample 2D bounding boxes for testing."""
    np.random.seed(42)
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 4) * 100
    boxes[:, 2] += boxes[:, 0] + 1  # xmax > xmin
    boxes[:, 3] += boxes[:, 1] + 1  # ymax > ymin
    return idx, boxes


@pytest.fixture
def sample_boxes_3d():
    """Generate sample 3D bounding boxes for testing."""
    np.random.seed(42)
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 6) * 100
    for i in range(3):
        boxes[:, i + 3] += boxes[:, i] + 1
    return idx, boxes


@pytest.fixture
def sample_boxes_4d():
    """Generate sample 4D bounding boxes for testing."""
    np.random.seed(42)
    n = 100
    idx = np.arange(n)
    boxes = np.random.rand(n, 8) * 100
    for i in range(4):
        boxes[:, i + 4] += boxes[:, i] + 1
    return idx, boxes


def has_intersect(x, y, dim):
    """Helper function to check if two boxes intersect."""
    return all([max(x[i], y[i]) <= min(x[i + dim], y[i + dim]) for i in range(dim)])
