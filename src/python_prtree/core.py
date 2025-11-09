"""Core PRTree classes for 2D, 3D, and 4D spatial indexing."""

import pickle
import numpy as np
from typing import Any, List, Optional, Sequence, Union

from .PRTree import (
    _PRTree2D_float32, _PRTree2D_float64,
    _PRTree3D_float32, _PRTree3D_float64,
    _PRTree4D_float32, _PRTree4D_float64,
)

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

    Automatically selects float32 or float64 precision based on input dtype.
    """

    Klass_float32 = None  # To be overridden by subclasses
    Klass_float64 = None  # To be overridden by subclasses

    def __init__(self, *args, **kwargs):
        """
        Initialize PRTree with optional indices and bounding boxes.

        Automatically selects precision based on input array dtype:
        - float32 input → float32 tree (native float32 precision)
        - float64 input → float64 tree (native double precision)
        - No input → float64 tree (default to higher precision)
        """
        if self.Klass_float32 is None or self.Klass_float64 is None:
            raise NotImplementedError("Use PRTree2D, PRTree3D, or PRTree4D")

        # Determine precision based on input
        use_float64 = True  # Default to float64 for empty constructor

        if len(args) >= 2:
            # Constructor with indices and boxes
            boxes = args[1]
            if hasattr(boxes, 'dtype'):
                # NumPy array - check dtype
                if boxes.dtype == np.float32:
                    use_float64 = False
                elif boxes.dtype == np.float64:
                    use_float64 = True
                else:
                    # Other types (int, etc.) - convert to float64 for safety
                    args = list(args)
                    args[1] = np.asarray(boxes, dtype=np.float64)
                    use_float64 = True
            else:
                # Convert to numpy array and default to float64
                args = list(args)
                args[1] = np.asarray(boxes, dtype=np.float64)
                use_float64 = True

        # Select appropriate class
        Klass = self.Klass_float64 if use_float64 else self.Klass_float32
        self._tree = Klass(*args, **kwargs)
        self._use_float64 = use_float64

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
                    Klass = self.Klass_float64 if self._use_float64 else self.Klass_float32
                    self._tree = Klass()
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

        # Convert bb to numpy array with appropriate dtype
        if not hasattr(bb, 'dtype'):
            # Convert to numpy array matching tree precision
            bb = np.asarray(bb, dtype=np.float64 if self._use_float64 else np.float32)

        objdumps = _dumps(obj)
        if self.n == 0:
            # Reinitialize tree with correct precision and preserve settings
            Klass = self.Klass_float64 if self._use_float64 else self.Klass_float32
            old_tree = self._tree

            # Check if subnormal detection is disabled - if so, use workaround
            subnormal_disabled = (hasattr(old_tree, 'get_subnormal_detection') and
                                  not old_tree.get_subnormal_detection())

            if subnormal_disabled:
                # Create with dummy valid box first
                dummy_idx = -999999
                dummy_bb = np.ones(len(bb), dtype=bb.dtype)
                self._tree = Klass([dummy_idx], [dummy_bb])

                # Preserve settings and disable subnormal detection
                if hasattr(old_tree, 'get_relative_epsilon'):
                    self._tree.set_relative_epsilon(old_tree.get_relative_epsilon())
                if hasattr(old_tree, 'get_absolute_epsilon'):
                    self._tree.set_absolute_epsilon(old_tree.get_absolute_epsilon())
                if hasattr(old_tree, 'get_adaptive_epsilon'):
                    self._tree.set_adaptive_epsilon(old_tree.get_adaptive_epsilon())
                self._tree.set_subnormal_detection(False)

                # Now insert the real box (tree is not empty, insert will work)
                self._tree.insert(idx, bb, objdumps)
                # Erase dummy
                self._tree.erase(dummy_idx)
            else:
                # Normal path
                self._tree = Klass([idx], [bb])

                # Preserve settings from old tree
                if hasattr(old_tree, 'get_relative_epsilon'):
                    self._tree.set_relative_epsilon(old_tree.get_relative_epsilon())
                if hasattr(old_tree, 'get_absolute_epsilon'):
                    self._tree.set_absolute_epsilon(old_tree.get_absolute_epsilon())
                if hasattr(old_tree, 'get_adaptive_epsilon'):
                    self._tree.set_adaptive_epsilon(old_tree.get_adaptive_epsilon())
                if hasattr(old_tree, 'get_subnormal_detection'):
                    self._tree.set_subnormal_detection(old_tree.get_subnormal_detection())

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
            if hasattr(queries, 'shape'):
                return [[] for _ in range(len(queries))]
            return []

        return self._tree.batch_query(queries, *args, **kwargs)


class PRTree2D(PRTreeBase):
    """
    2D Priority R-Tree for spatial indexing.

    Supports efficient querying of 2D bounding boxes:
    [xmin, ymin, xmax, ymax]

    Automatically uses float32 or float64 precision based on input dtype.

    Example:
        >>> # Float64 precision (default)
        >>> tree = PRTree2D([1, 2], [[0, 0, 1, 1], [2, 2, 3, 3]])
        >>>
        >>> # Explicit float32 precision
        >>> import numpy as np
        >>> tree_f32 = PRTree2D([1, 2], np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32))
        >>>
        >>> results = tree.query([0.5, 0.5, 2.5, 2.5])
        >>> print(results)  # [1, 2]
    """
    Klass_float32 = _PRTree2D_float32
    Klass_float64 = _PRTree2D_float64


class PRTree3D(PRTreeBase):
    """
    3D Priority R-Tree for spatial indexing.

    Supports efficient querying of 3D bounding boxes:
    [xmin, ymin, zmin, xmax, ymax, zmax]

    Automatically uses float32 or float64 precision based on input dtype.

    Example:
        >>> tree = PRTree3D([1], [[0, 0, 0, 1, 1, 1]])
        >>> results = tree.query([0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    """
    Klass_float32 = _PRTree3D_float32
    Klass_float64 = _PRTree3D_float64


class PRTree4D(PRTreeBase):
    """
    4D Priority R-Tree for spatial indexing.

    Supports efficient querying of 4D bounding boxes.
    Useful for spatio-temporal data or higher-dimensional spaces.

    Automatically uses float32 or float64 precision based on input dtype.
    """
    Klass_float32 = _PRTree4D_float32
    Klass_float64 = _PRTree4D_float64
