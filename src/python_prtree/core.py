"""Core PRTree classes for 2D, 3D, and 4D spatial indexing."""

import pickle
from typing import Any, List, Optional, Sequence, Union

from .PRTree import _PRTree2D, _PRTree3D, _PRTree4D

__all__ = [
    "PRTree2D",
    "PRTree3D",
    "PRTree4D",
]


def _dumps(obj: Any) -> Optional[bytes]:
    """Serialize Python object using pickle."""
    if obj is None:
        return None
    return pickle.dumps(obj)


def _loads(obj: Optional[bytes]) -> Any:
    """Deserialize Python object using pickle."""
    if obj is None:
        return None
    return pickle.loads(obj)


class PRTreeBase:
    """
    Base class for PRTree implementations.

    Provides common functionality for 2D, 3D, and 4D spatial indexing
    with Priority R-Tree data structure.
    """

    Klass = None  # To be overridden by subclasses

    def __init__(self, *args, **kwargs):
        """Initialize PRTree with optional indices and bounding boxes."""
        if self.Klass is None:
            raise NotImplementedError("Use PRTree2D, PRTree3D, or PRTree4D")
        self._tree = self.Klass(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to underlying C++ tree."""
        def handler_function(*args, **kwargs):
            # Handle empty tree cases for methods that cause segfaults
            if self.n == 0 and name in ('rebuild', 'save'):
                # These operations are not meaningful/safe on empty trees
                if name == 'rebuild':
                    return  # No-op for empty tree
                elif name == 'save':
                    raise ValueError("Cannot save empty tree")

            ret = getattr(self._tree, name)(*args, **kwargs)
            return ret

        return handler_function

    @property
    def n(self) -> int:
        """Get the number of bounding boxes in the tree."""
        return self._tree.size()

    def __len__(self) -> int:
        """Return the number of bounding boxes in the tree."""
        return self.n

    def erase(self, idx: int) -> None:
        """
        Remove a bounding box by index.

        Args:
            idx: Index of the bounding box to remove

        Raises:
            ValueError: If tree is empty or index not found
        """
        if self.n == 0:
            raise ValueError("Nothing to erase")

        # Handle erasing the last element (library limitation workaround)
        if self.n == 1:
            # Call underlying erase to validate index, then handle the library bug
            try:
                self._tree.erase(idx)
                # If we get here, erase succeeded (shouldn't happen with n==1)
                return
            except RuntimeError as e:
                error_msg = str(e)
                if "Given index is not found" in error_msg:
                    # Index doesn't exist - re-raise the error
                    raise
                elif "#roots is not 1" in error_msg:
                    # This is the library bug we're working around
                    # Index was valid, so recreate empty tree
                    self._tree = self.Klass()
                    return
                else:
                    # Some other RuntimeError - re-raise it
                    raise

        self._tree.erase(idx)

    def set_obj(self, idx: int, obj: Any) -> None:
        """
        Store a Python object associated with a bounding box.

        Args:
            idx: Index of the bounding box
            obj: Any picklable Python object
        """
        objdumps = _dumps(obj)
        self._tree.set_obj(idx, objdumps)

    def get_obj(self, idx: int) -> Any:
        """
        Retrieve the Python object associated with a bounding box.

        Args:
            idx: Index of the bounding box

        Returns:
            The stored Python object, or None if not set
        """
        obj = self._tree.get_obj(idx)
        return _loads(obj)

    def insert(
        self,
        idx: Optional[int] = None,
        bb: Optional[Sequence[float]] = None,
        obj: Any = None
    ) -> None:
        """
        Insert a new bounding box into the tree.

        Args:
            idx: Index for the bounding box (auto-assigned if None)
            bb: Bounding box coordinates (required)
            obj: Optional Python object to associate

        Raises:
            ValueError: If bounding box is not specified
        """
        if idx is None and obj is None:
            raise ValueError("Specify index or obj")
        if idx is None:
            idx = self.n + 1
        if bb is None:
            raise ValueError("Specify bounding box")

        objdumps = _dumps(obj)
        if self.n == 0:
            self._tree = self.Klass([idx], [bb])
            self._tree.set_obj(idx, objdumps)
        else:
            self._tree.insert(idx, bb, objdumps)

    def query(
        self,
        *args,
        return_obj: bool = False
    ) -> Union[List[int], List[Any]]:
        """
        Find all bounding boxes that overlap with the query box.

        Args:
            *args: Query bounding box coordinates
            return_obj: If True, return stored objects instead of indices

        Returns:
            List of indices or objects that overlap with the query
        """
        # Handle empty tree case to prevent segfault
        if self.n == 0:
            return []

        if len(args) == 1:
            out = self._tree.query(*args)
        else:
            out = self._tree.query(args)

        if return_obj:
            objs = [self.get_obj(i) for i in out]
            return objs
        else:
            return out

    def batch_query(self, queries, *args, **kwargs):
        """
        Perform multiple queries in parallel.

        Args:
            queries: Array of query bounding boxes
            *args, **kwargs: Additional arguments passed to C++ implementation

        Returns:
            List of result lists, one per query
        """
        # Handle empty tree case to prevent segfault
        if self.n == 0:
            # Return empty list for each query
            import numpy as np
            if hasattr(queries, 'shape'):
                return [[] for _ in range(len(queries))]
            return []

        return self._tree.batch_query(queries, *args, **kwargs)


class PRTree2D(PRTreeBase):
    """
    2D Priority R-Tree for spatial indexing.

    Supports efficient querying of 2D bounding boxes:
    [xmin, ymin, xmax, ymax]

    Example:
        >>> tree = PRTree2D([1, 2], [[0, 0, 1, 1], [2, 2, 3, 3]])
        >>> results = tree.query([0.5, 0.5, 2.5, 2.5])
        >>> print(results)  # [1, 2]
    """
    Klass = _PRTree2D


class PRTree3D(PRTreeBase):
    """
    3D Priority R-Tree for spatial indexing.

    Supports efficient querying of 3D bounding boxes:
    [xmin, ymin, zmin, xmax, ymax, zmax]

    Example:
        >>> tree = PRTree3D([1], [[0, 0, 0, 1, 1, 1]])
        >>> results = tree.query([0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    """
    Klass = _PRTree3D


class PRTree4D(PRTreeBase):
    """
    4D Priority R-Tree for spatial indexing.

    Supports efficient querying of 4D bounding boxes.
    Useful for spatio-temporal data or higher-dimensional spaces.
    """
    Klass = _PRTree4D
