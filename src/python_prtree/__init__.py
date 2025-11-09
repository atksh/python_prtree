"""
python_prtree - Fast spatial indexing with Priority R-Tree

This package provides efficient 2D, 3D, and 4D spatial indexing using
the Priority R-Tree data structure with C++ performance.

Main classes:
    - PRTree2D: 2D spatial indexing
    - PRTree3D: 3D spatial indexing
    - PRTree4D: 4D spatial indexing

Example:
    >>> from python_prtree import PRTree2D
    >>> import numpy as np
    >>>
    >>> # Create tree with bounding boxes
    >>> indices = np.array([1, 2, 3])
    >>> boxes = np.array([
    ...     [0.0, 0.0, 1.0, 1.0],
    ...     [1.0, 1.0, 2.0, 2.0],
    ...     [2.0, 2.0, 3.0, 3.0],
    ... ])
    >>> tree = PRTree2D(indices, boxes)
    >>>
    >>> # Query overlapping boxes
    >>> results = tree.query([0.5, 0.5, 1.5, 1.5])
    >>> print(results)  # [1, 2]

For more information, see the documentation at:
https://github.com/atksh/python_prtree
"""

from .core import PRTree2D, PRTree3D, PRTree4D

__version__ = "0.7.1"

__all__ = [
    "PRTree2D",
    "PRTree3D",
    "PRTree4D",
]
