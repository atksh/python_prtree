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
    Base class for PRTree implementations providing spatial indexing.

    PRTreeBase implements the Priority R-Tree data structure for efficient
    spatial querying of bounding boxes in 2D, 3D, or 4D space. This base
    class provides common functionality shared across all dimensions.

    The implementation automatically selects between float32 and float64
    precision based on the input data type, ensuring optimal performance
    while maintaining numerical accuracy.

    Attributes:
        Klass_float32: C++ binding class for float32 precision (set by subclasses)
        Klass_float64: C++ binding class for float64 precision (set by subclasses)
        _tree: Underlying C++ tree instance
        _use_float64: Boolean flag indicating current precision level

    Thread Safety:
        - Read operations (query, batch_query) are thread-safe
        - Write operations (insert, erase, rebuild) require external synchronization
        - Do not mix read and write operations without proper locking

    See Also:
        PRTree2D: 2D spatial indexing implementation
        PRTree3D: 3D spatial indexing implementation
        PRTree4D: 4D spatial indexing implementation
    """

    Klass_float32 = None  # To be overridden by subclasses
    Klass_float64 = None  # To be overridden by subclasses

    def __init__(self, *args, **kwargs):
        """
        Initialize Priority R-Tree with optional data or load from file.

        This constructor supports three modes of initialization:
        1. Empty tree: PRTree() - creates an empty tree with float64 precision
        2. With data: PRTree(indices, boxes) - builds tree from arrays
        3. From file: PRTree(filename) - loads previously saved tree

        Precision is automatically selected based on input:
        - float32 input → native float32 precision tree
        - float64 input → native float64 (double) precision tree
        - Other types → converted to float64 for safety
        - No input → defaults to float64 for higher precision
        - From file → precision auto-detected from saved data

        Args:
            *args: Variable length argument list:
                - Empty: no arguments for empty tree
                - Data: (indices, boxes) where:
                    - indices: array-like of integers, shape (n,)
                    - boxes: array-like of floats, shape (n, 2*D) where D is dimension
                - File: single string argument with file path
            **kwargs: Additional keyword arguments passed to C++ implementation

        Raises:
            NotImplementedError: If called directly on base class (use PRTree2D/3D/4D)
            ValueError: If file cannot be loaded or has unsupported format

        Examples:
            >>> # Empty tree
            >>> tree = PRTree2D()

            >>> # With data (float64 precision)
            >>> indices = np.array([1, 2, 3])
            >>> boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]])
            >>> tree = PRTree2D(indices, boxes)

            >>> # With float32 precision
            >>> boxes_f32 = np.array([[0, 0, 1, 1]], dtype=np.float32)
            >>> tree = PRTree2D([1], boxes_f32)

            >>> # Load from file
            >>> tree = PRTree2D('saved_tree.bin')

        Note:
            Precision selection affects both memory usage and numerical accuracy.
            Float32 uses less memory but may have reduced precision for very
            large coordinate values or small distances.
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
        elif len(args) == 1 and isinstance(args[0], str):
            # Loading from file - try both precisions to auto-detect
            filepath = args[0]

            # Try float32 first (more common for saved files)
            try:
                self._tree = self.Klass_float32(filepath, **kwargs)
                self._use_float64 = False
            except Exception:
                # If float32 fails, try float64
                try:
                    self._tree = self.Klass_float64(filepath, **kwargs)
                    self._use_float64 = True
                except Exception as e:
                    # Both failed - raise informative error
                    raise ValueError(f"Failed to load tree from {filepath}. "
                                   f"File may be corrupted or in unsupported format.") from e
        else:
            # Empty constructor or other cases - default to float64
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
        Remove a bounding box from the tree by its index.

        This method removes the bounding box with the specified index from
        the spatial index. The operation modifies the tree structure and
        requires O(log n) time in the average case.

        Important: This is a write operation that modifies the tree. If using
        in a multi-threaded environment, ensure proper external synchronization
        to prevent concurrent access with read operations.

        Args:
            idx (int): Index of the bounding box to remove. Must be an index
                      that was previously inserted into the tree.

        Raises:
            ValueError: If the tree is empty (no elements to erase)
            RuntimeError: If the specified index is not found in the tree

        Examples:
            >>> tree = PRTree2D([1, 2, 3], [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]])
            >>> tree.size()
            3
            >>> tree.erase(2)
            >>> tree.size()
            2
            >>> tree.query([2, 2, 3, 3])  # Box with index 2 no longer found
            []

        Note:
            After multiple erase operations, consider calling rebuild() to
            optimize tree structure and query performance.

        Thread Safety:
            Not thread-safe with concurrent read or write operations.
            Use external locking if needed.

        See Also:
            insert: Add a new bounding box to the tree
            rebuild: Rebuild tree structure for optimal performance
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
        Store or update a Python object associated with a bounding box.

        This method associates an arbitrary Python object with a bounding box
        in the tree. The object must be picklable and will be serialized for
        storage. This allows attaching metadata, application-specific data,
        or any Python object to spatial elements.

        Args:
            idx (int): Index of the bounding box to associate the object with.
                      The index must exist in the tree (previously inserted).
            obj (Any): Any picklable Python object to store. Common examples:
                      - dict: {"name": "Building A", "height": 100}
                      - str: "Building A"
                      - Custom objects: MyClass() (if picklable)
                      - None: Remove/clear the associated object

        Raises:
            RuntimeError: If the index does not exist in the tree
            TypeError: If the object is not picklable

        Examples:
            >>> tree = PRTree2D([1, 2], [[0, 0, 1, 1], [2, 2, 3, 3]])

            >>> # Associate dict objects
            >>> tree.set_obj(1, {"name": "Building A", "floors": 10})
            >>> tree.set_obj(2, {"name": "Building B", "floors": 20})

            >>> # Retrieve during query
            >>> results = tree.query([0, 0, 3, 3], return_obj=True)
            >>> print(results)
            [{'name': 'Building A', 'floors': 10}, {'name': 'Building B', 'floors': 20}]

            >>> # Update existing object
            >>> tree.set_obj(1, {"name": "Building A - Renovated", "floors": 12})

            >>> # Clear object
            >>> tree.set_obj(2, None)

        Note:
            Objects are serialized using pickle, which adds storage overhead.
            For large numbers of small objects, consider storing a reference
            (like an ID) instead of the full object.

        Thread Safety:
            This operation modifies internal state. Use external synchronization
            if concurrent access is needed.

        See Also:
            get_obj: Retrieve the object associated with an index
            insert: Insert a bounding box with an associated object
            query: Query with return_obj=True to get objects directly
        """
        objdumps = _dumps(obj)
        self._tree.set_obj(idx, objdumps)

    def get_obj(self, idx: int) -> Any:
        """
        Retrieve the Python object associated with a bounding box.

        This method retrieves the Python object that was associated with a
        bounding box using set_obj() or insert(obj=...). The object is
        deserialized from its pickled form.

        Args:
            idx (int): Index of the bounding box whose object to retrieve.
                      The index must exist in the tree.

        Returns:
            Any: The Python object associated with this index, or None if:
                 - No object was associated with this index
                 - The object was explicitly set to None
                 - The box was inserted without an object

        Raises:
            RuntimeError: If the index does not exist in the tree

        Examples:
            >>> tree = PRTree2D()
            >>> tree.insert(idx=1, bb=[0, 0, 1, 1], obj={"type": "building"})
            >>> tree.insert(idx=2, bb=[2, 2, 3, 3])  # No object

            >>> # Retrieve object
            >>> obj1 = tree.get_obj(1)
            >>> print(obj1)
            {'type': 'building'}

            >>> # No object was set
            >>> obj2 = tree.get_obj(2)
            >>> print(obj2)
            None

            >>> # Alternative: use query with return_obj=True
            >>> results = tree.query([0, 0, 3, 3], return_obj=True)
            >>> print(results)  # Both objects in query order
            [{'type': 'building'}, None]

        Performance:
            Object retrieval requires deserialization (unpickling), which may
            be slower for large or complex objects. For high-performance
            scenarios, consider storing lightweight references instead.

        Thread Safety:
            Read-only operation, thread-safe with concurrent get_obj() calls.
            Do not call concurrently with set_obj() without synchronization.

        See Also:
            set_obj: Store an object associated with an index
            insert: Insert a bounding box with an associated object
            query: Query with return_obj=True to get objects in batch
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
        Insert a new bounding box into the tree with optional associated object.

        This method adds a new bounding box to the spatial index. The box can
        optionally be associated with a Python object for later retrieval.
        If no index is provided and obj is given, an auto-incremented index
        will be assigned.

        The bounding box coordinates must follow the format:
        - 2D: [xmin, ymin, xmax, ymax]
        - 3D: [xmin, ymin, zmin, xmax, ymax, zmax]
        - 4D: [x1min, x2min, x3min, x4min, x1max, x2max, x3max, x4max]

        All coordinates must satisfy min <= max for each dimension.

        Important: This is a write operation that modifies the tree. If using
        in a multi-threaded environment, ensure proper external synchronization
        to prevent concurrent access with read operations.

        Args:
            idx (Optional[int]): Index for the bounding box. If None, will be
                                auto-assigned starting from (current_size + 1).
                                Must be unique.
            bb (Optional[Sequence[float]]): Bounding box coordinates as a sequence.
                                           Required. Length must be 2*D where D
                                           is the tree dimension (2, 3, or 4).
            obj (Any): Optional Python object to associate with this bounding box.
                      Must be picklable. Can be retrieved later using get_obj()
                      or by setting return_obj=True in query().

        Raises:
            ValueError: If bb is None (bounding box must be specified)
            ValueError: If both idx and obj are None (at least one must be specified)
            RuntimeError: If coordinates are invalid (min > max for any dimension)
            RuntimeError: If index already exists in the tree

        Examples:
            >>> tree = PRTree2D()

            >>> # Insert with explicit index
            >>> tree.insert(idx=1, bb=[0, 0, 1, 1])

            >>> # Insert with auto-assigned index and object
            >>> tree.insert(bb=[2, 2, 3, 3], obj={"name": "Building A"})

            >>> # Insert with both index and object
            >>> tree.insert(idx=10, bb=[4, 4, 5, 5], obj={"name": "Building B"})

            >>> # Query and retrieve objects
            >>> results = tree.query([0, 0, 5, 5], return_obj=True)
            >>> print(results)
            [None, {'name': 'Building A'}, {'name': 'Building B'}]

        Note:
            After multiple insert operations, consider calling rebuild() to
            optimize tree structure and query performance.

            The precision (float32/float64) of the inserted bounding box will
            be automatically converted to match the tree's precision.

        Thread Safety:
            Not thread-safe with concurrent read or write operations.
            Use external locking if needed.

        See Also:
            erase: Remove a bounding box from the tree
            rebuild: Rebuild tree structure for optimal performance
            get_obj: Retrieve the object associated with an index
            set_obj: Update the object associated with an index
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
        Find all bounding boxes that overlap with the query box or point.

        This method performs a spatial query to find all bounding boxes in the
        tree that intersect with the given query region. The query can be either
        a bounding box or a point (which is treated as a box with zero volume).

        The intersection test uses closed intervals, meaning boxes that touch
        at their boundaries are considered intersecting.

        This is a read-only operation and is thread-safe when used concurrently
        from multiple threads, as long as no write operations (insert/erase/rebuild)
        are being performed simultaneously.

        Args:
            *args: Query coordinates in one of these formats:
                - Array/list: [min1, min2, ..., max1, max2, ...] for box query
                - Array/list: [coord1, coord2, ...] for point query
                - Varargs: query(min1, min2, ..., max1, max2, ...) for box
                - Varargs: query(coord1, coord2, ...) for point
                The number of coordinates must match the tree dimension:
                - 2D: 4 values for box [xmin, ymin, xmax, ymax] or 2 for point [x, y]
                - 3D: 6 values for box [xmin, ymin, zmin, xmax, ymax, zmax] or 3 for point
                - 4D: 8 values for box or 4 for point
            return_obj (bool): If True, return the Python objects associated with
                              each bounding box instead of indices. Default is False.

        Returns:
            Union[List[int], List[Any]]:
                - If return_obj=False: List of integer indices of overlapping boxes
                - If return_obj=True: List of associated Python objects (may contain None)
                Empty list if no overlaps found or tree is empty.

        Raises:
            RuntimeError: If query coordinates have invalid shape/length

        Examples:
            >>> tree = PRTree2D([1, 2, 3], [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]])

            >>> # Box query with list
            >>> tree.query([0.5, 0.5, 2.5, 2.5])
            [1, 2]

            >>> # Box query with varargs
            >>> tree.query(0.5, 0.5, 2.5, 2.5)
            [1, 2]

            >>> # Point query
            >>> tree.query([0.5, 0.5])
            [1]

            >>> # Point query with varargs
            >>> tree.query(0.5, 0.5)
            [1]

            >>> # Query with objects
            >>> tree2 = PRTree2D()
            >>> tree2.insert(bb=[0, 0, 1, 1], obj={"name": "Box A"})
            >>> tree2.insert(bb=[2, 2, 3, 3], obj={"name": "Box B"})
            >>> tree2.query([0.5, 0.5, 2.5, 2.5], return_obj=True)
            [{'name': 'Box A'}, {'name': 'Box B'}]

        Performance:
            Query time complexity is O(log n + k) where n is the total number
            of boxes and k is the number of results returned.

        Thread Safety:
            Thread-safe for concurrent queries. Do not call during write operations
            (insert/erase/rebuild) without external synchronization.

        See Also:
            batch_query: Parallel queries for multiple query boxes
            query_intersections: Find all pairs of intersecting boxes in the tree
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
        Perform multiple spatial queries in parallel for high performance.

        This method executes multiple queries simultaneously using C++ std::thread
        for parallelization. It automatically utilizes multiple CPU cores for
        significant speedup compared to sequential single queries.

        The queries are distributed across threads based on hardware_concurrency(),
        making this method ideal for processing large batches of spatial queries.

        This is a read-only operation and is thread-safe when used concurrently
        from multiple threads, as long as no write operations are being performed.

        Args:
            queries (array-like): Array of query bounding boxes, shape (n_queries, 2*D)
                                 where D is the tree dimension. Each row represents
                                 one query box with format:
                                 - 2D: [xmin, ymin, xmax, ymax]
                                 - 3D: [xmin, ymin, zmin, xmax, ymax, zmax]
                                 - 4D: [x1min, x2min, x3min, x4min, x1max, x2max, x3max, x4max]
            *args: Additional positional arguments (passed to C++ implementation)
            **kwargs: Additional keyword arguments (passed to C++ implementation)

        Returns:
            List[List[int]]: List of result lists, one per input query.
                            Each inner list contains the indices of bounding boxes
                            that overlap with the corresponding query box.
                            Returns list of empty lists if tree is empty.

        Examples:
            >>> tree = PRTree2D([1, 2, 3], [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]])

            >>> # Multiple box queries
            >>> queries = np.array([
            ...     [0.5, 0.5, 1.5, 1.5],  # Query 1
            ...     [2.5, 2.5, 4.5, 4.5],  # Query 2
            ...     [0, 0, 5, 5],          # Query 3
            ... ])
            >>> results = tree.batch_query(queries)
            >>> print(results)
            [[1], [2, 3], [1, 2, 3]]

            >>> # Single query (note: returns list of lists)
            >>> single_query = np.array([[0, 0, 1, 1]])
            >>> results = tree.batch_query(single_query)
            >>> print(results)  # [[1]] - list containing one result list
            [[1]]

        Performance:
            - Automatically parallelized using all available CPU cores
            - Ideal for batches of 100+ queries where parallelization overhead is amortized
            - For small batches (<10 queries), sequential query() may be faster
            - Time complexity: O((log n + k) * m / p) where:
              - n = number of boxes in tree
              - k = average number of results per query
              - m = number of queries
              - p = number of parallel threads

        Thread Safety:
            Thread-safe for concurrent batch queries from Python threads.
            Internal C++ parallelization is independent of Python threading.
            Do not call during write operations without external synchronization.

        Note:
            batch_query internally uses C++ std::thread for parallelization,
            which is independent of Python's GIL (Global Interpreter Lock).
            This provides true parallel execution even in CPython.

        See Also:
            query: Single spatial query
            query_intersections: Find all pairs of intersecting boxes
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
    2D Priority R-Tree for efficient spatial indexing of 2D bounding boxes.

    PRTree2D provides fast spatial queries for 2D rectangles using the
    Priority R-Tree data structure. It excels at finding all rectangles
    that overlap with a query region, making it ideal for GIS applications,
    collision detection, and spatial databases.

    Bounding Box Format:
        Each 2D bounding box is represented as [xmin, ymin, xmax, ymax]
        where xmin <= xmax and ymin <= ymax.

    Precision:
        Automatically selects between float32 and float64 precision based
        on input numpy array dtype. Float32 uses less memory while float64
        provides higher numerical accuracy.

    Performance:
        - Construction: O(n log n) for n boxes
        - Query: O(log n + k) where k is number of results
        - Insert/Erase: O(log n) amortized
        - Batch query: Parallelized across CPU cores

    Thread Safety:
        - Read operations (query, batch_query): Thread-safe
        - Write operations (insert, erase): Require external synchronization
        - Do NOT mix reads and writes without locking

    Attributes:
        n (int): Number of bounding boxes in the tree (property)
        Klass_float32: C++ class for float32 precision
        Klass_float64: C++ class for float64 precision

    Examples:
        Basic usage:
            >>> import numpy as np
            >>> from python_prtree import PRTree2D
            >>>
            >>> # Create tree with bounding boxes
            >>> indices = np.array([1, 2, 3])
            >>> boxes = np.array([
            ...     [0.0, 0.0, 1.0, 1.0],  # Box 1
            ...     [2.0, 2.0, 3.0, 3.0],  # Box 2
            ...     [1.5, 1.5, 2.5, 2.5],  # Box 3
            ... ])
            >>> tree = PRTree2D(indices, boxes)
            >>>
            >>> # Query overlapping boxes
            >>> results = tree.query([0.5, 0.5, 2.5, 2.5])
            >>> print(results)  # [1, 3]
            >>>
            >>> # Batch query (parallel)
            >>> queries = np.array([[0, 0, 1, 1], [2, 2, 3, 3]])
            >>> results = tree.batch_query(queries)
            >>> print(results)  # [[1], [2, 3]]

        With float32 precision:
            >>> boxes_f32 = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32)
            >>> tree_f32 = PRTree2D([1, 2], boxes_f32)  # Uses float32 internally

        With Python objects:
            >>> tree = PRTree2D()
            >>> tree.insert(bb=[0, 0, 1, 1], obj={"name": "Building A"})
            >>> tree.insert(bb=[2, 2, 3, 3], obj={"name": "Building B"})
            >>> results = tree.query([0, 0, 3, 3], return_obj=True)
            >>> print(results)  # [{'name': 'Building A'}, {'name': 'Building B'}]

        Save and load:
            >>> tree.save('spatial_index.bin')
            >>> loaded_tree = PRTree2D('spatial_index.bin')

    See Also:
        PRTree3D: 3D spatial indexing
        PRTree4D: 4D spatial indexing

    References:
        Priority R-Tree: Arge et al., SIGMOD 2004
        https://www.cse.ust.hk/~yike/prtree/
    """
    Klass_float32 = _PRTree2D_float32
    Klass_float64 = _PRTree2D_float64


class PRTree3D(PRTreeBase):
    """
    3D Priority R-Tree for efficient spatial indexing of 3D bounding boxes.

    PRTree3D provides fast spatial queries for 3D axis-aligned bounding boxes
    (AABBs) using the Priority R-Tree data structure. It is ideal for 3D
    applications such as collision detection in games, volumetric data
    analysis, and 3D GIS.

    Bounding Box Format:
        Each 3D bounding box is represented as [xmin, ymin, zmin, xmax, ymax, zmax]
        where xmin <= xmax, ymin <= ymax, and zmin <= zmax.

    Precision:
        Automatically selects between float32 and float64 precision based
        on input numpy array dtype.

    Performance:
        Same asymptotic complexity as PRTree2D, optimized for 3D operations.

    Thread Safety:
        Same as PRTree2D - reads are thread-safe, writes require synchronization.

    Examples:
        Basic usage:
            >>> import numpy as np
            >>> from python_prtree import PRTree3D
            >>>
            >>> # Create tree with 3D bounding boxes
            >>> indices = np.array([1, 2])
            >>> boxes = np.array([
            ...     [0, 0, 0, 1, 1, 1],  # Cube 1
            ...     [2, 2, 2, 3, 3, 3],  # Cube 2
            ... ])
            >>> tree = PRTree3D(indices, boxes)
            >>>
            >>> # Query overlapping boxes
            >>> results = tree.query([0.5, 0.5, 0.5, 2.5, 2.5, 2.5])
            >>> print(results)  # [1]
            >>>
            >>> # Point query
            >>> results = tree.query([0.5, 0.5, 0.5])  # Point inside cube 1
            >>> print(results)  # [1]

    See Also:
        PRTree2D: 2D spatial indexing
        PRTree4D: 4D spatial indexing
    """
    Klass_float32 = _PRTree3D_float32
    Klass_float64 = _PRTree3D_float64


class PRTree4D(PRTreeBase):
    """
    4D Priority R-Tree for efficient spatial indexing of 4D bounding boxes.

    PRTree4D provides fast spatial queries for 4D axis-aligned bounding boxes
    using the Priority R-Tree data structure. This is particularly useful for:
    - Spatio-temporal data (3D space + time dimension)
    - Higher-dimensional feature spaces
    - Multi-parameter range queries

    Bounding Box Format:
        Each 4D bounding box is represented as:
        [x1min, x2min, x3min, x4min, x1max, x2max, x3max, x4max]
        where ximin <= ximax for each dimension i.

    Precision:
        Automatically selects between float32 and float64 precision based
        on input numpy array dtype.

    Performance:
        Same asymptotic complexity as PRTree2D/3D, optimized for 4D operations.
        Note that higher dimensions naturally have larger search spaces.

    Thread Safety:
        Same as PRTree2D - reads are thread-safe, writes require synchronization.

    Examples:
        Basic usage:
            >>> import numpy as np
            >>> from python_prtree import PRTree4D
            >>>
            >>> # Create tree with 4D bounding boxes
            >>> # Example: 3D space (x,y,z) + time (t)
            >>> indices = np.array([1, 2])
            >>> boxes = np.array([
            ...     [0, 0, 0, 0, 1, 1, 1, 10],  # Event 1: space [0,1]³, time [0,10]
            ...     [2, 2, 2, 5, 3, 3, 3, 15],  # Event 2: space [2,3]³, time [5,15]
            ... ])
            >>> tree = PRTree4D(indices, boxes)
            >>>
            >>> # Query: find events in space [0,2.5]³ during time [0,7]
            >>> results = tree.query([0, 0, 0, 0, 2.5, 2.5, 2.5, 7])
            >>> print(results)  # [1]

        Spatio-temporal query:
            >>> # Find all objects present at location (0.5, 0.5, 0.5) at time 5
            >>> results = tree.query([0.5, 0.5, 0.5, 5])  # Point query
            >>> print(results)  # [1]

    Use Cases:
        - Spatio-temporal databases (trajectories, events)
        - Video analysis (x, y, frame, feature)
        - Multi-dimensional parameter spaces
        - Time-series spatial data

    See Also:
        PRTree2D: 2D spatial indexing
        PRTree3D: 3D spatial indexing
    """
    Klass_float32 = _PRTree4D_float32
    Klass_float64 = _PRTree4D_float64
