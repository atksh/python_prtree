# python_prtree

Fast spatial indexing with Priority R-Tree for Python. Efficiently query 2D/3D/4D bounding boxes with C++ performance.

## Quick Start

### Installation

```bash
pip install python-prtree
```

### Basic Usage

```python
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
print(result)  # [1]

# Batch query (faster for multiple queries)
queries = np.array([
    [0.5, 0.2, 0.6, 0.3],
    [0.8, 0.5, 1.5, 3.5],
])
results = tree.batch_query(queries)
print(results)  # [[1], [1, 2]]
```

## Core Features

### Supported Operations

- **Construction**: Create from numpy arrays (2D, 3D, or 4D)
- **Query**: Find overlapping bounding boxes
- **Batch Query**: Parallel queries for high performance
- **Insert/Erase**: Dynamic updates (optimized for mostly static data)
- **Query Intersections**: Find all pairs of intersecting boxes
- **Save/Load**: Serialize tree to disk

### Supported Dimensions

```python
from python_prtree import PRTree2D, PRTree3D, PRTree4D

tree2d = PRTree2D(indices, boxes_2d)  # [xmin, ymin, xmax, ymax]
tree3d = PRTree3D(indices, boxes_3d)  # [xmin, ymin, zmin, xmax, ymax, zmax]
tree4d = PRTree4D(indices, boxes_4d)  # 4D boxes
```

## Usage Examples

### Point Queries

```python
# Query with point coordinates
result = tree.query([0.5, 0.5])        # Returns indices
result = tree.query(0.5, 0.5)          # Varargs also supported (2D only)
```

### Dynamic Updates

```python
# Insert new rectangle
tree.insert(3, np.array([1.0, 1.0, 2.0, 2.0]))

# Remove rectangle by index
tree.erase(2)

# Rebuild for optimal performance after many updates
tree.rebuild()
```

### Store Python Objects

```python
# Store any picklable Python object with rectangles
tree = PRTree2D()
tree.insert(bb=[0, 0, 1, 1], obj={"name": "Building A", "height": 100})
tree.insert(bb=[2, 2, 3, 3], obj={"name": "Building B", "height": 200})

# Query and retrieve objects
results = tree.query([0.5, 0.5, 2.5, 2.5], return_obj=True)
print(results)  # [{'name': 'Building A', 'height': 100}, {'name': 'Building B', 'height': 200}]
```

### Find Intersecting Pairs

```python
# Find all pairs of intersecting rectangles
pairs = tree.query_intersections()
print(pairs)  # numpy array of shape (n_pairs, 2)
# [[1, 3], [2, 5], ...]  # pairs of indices that intersect
```

### Save and Load

```python
# Save tree to file
tree.save('spatial_index.bin')

# Load from file
tree = PRTree2D('spatial_index.bin')

# Or load later
tree = PRTree2D()
tree.load('spatial_index.bin')
```

**Note**: Binary format may change between versions. Rebuild your tree after upgrading.

## Performance

### When to Use

✅ **Good for:**
- Large static datasets (millions of boxes)
- Batch queries (parallel processing)
- Spatial indexing, collision detection
- GIS applications, game engines

⚠️ **Not ideal for:**
- Frequent insertions/deletions (rebuild overhead)
- Real-time dynamic scenes with constant updates

### Benchmarks

Fast construction and query performance compared to alternatives:

#### Construction Time (2D)
![2d_construction](https://raw.githubusercontent.com/atksh/python_prtree/main/docs/images/2d_fig1.png)

#### Query Performance (2D)
![2d_query](https://raw.githubusercontent.com/atksh/python_prtree/main/docs/images/2d_fig2.png)

*Batch queries use parallel processing for significant speedup.*

## Important Notes

### Coordinate Format

Boxes must have **min ≤ max** for each dimension:
```python
# Correct
tree.insert(1, [0, 0, 1, 1])  # xmin=0 < xmax=1, ymin=0 < ymax=1

# Wrong - will raise error
tree.insert(1, [1, 1, 0, 0])  # xmin > xmax, ymin > ymax
```

### Empty Trees

All operations are safe on empty trees:
```python
tree = PRTree2D()
result = tree.query([0, 0, 1, 1])  # Returns []
results = tree.batch_query(queries)  # Returns [[], [], ...]
```

### Precision

- **Float32 input**: Pure float32 for maximum speed
- **Float64 input**: Float32 tree + double-precision refinement for accuracy
- Handles boxes with very small gaps correctly (< 1e-5)

### Thread Safety

- Query operations are thread-safe
- Insert/erase operations are NOT thread-safe
- Use external synchronization for concurrent updates

## Installation from Source

```bash
# Install dependencies
pip install -U cmake pybind11 numpy

# Clone with submodules
git clone --recursive https://github.com/atksh/python_prtree
cd python_prtree

# Build and install
python setup.py install
```

## API Reference

### PRTree2D / PRTree3D / PRTree4D

#### Constructor
```python
PRTree2D(indices=None, boxes=None)
PRTree2D(filename)  # Load from file
```

#### Methods
- `query(box, return_obj=False)` - Find overlapping boxes
- `batch_query(boxes)` - Parallel batch queries
- `query_intersections()` - Find all intersecting pairs
- `insert(idx, bb, obj=None)` - Add box
- `erase(idx)` - Remove box
- `rebuild()` - Rebuild tree for optimal performance
- `save(filename)` - Save to binary file
- `load(filename)` - Load from binary file
- `size()` - Get number of boxes
- `get_obj(idx)` - Get stored object
- `set_obj(idx, obj)` - Update stored object

## Version History

### v0.7.0 (Latest)
- **Fixed critical bug**: Boxes with small gaps (<1e-5) incorrectly reported as intersecting
- **Breaking**: Minimum Python 3.8, serialization format changed
- Added input validation (NaN/Inf rejection)
- Improved precision handling

### v0.5.x
- Added 4D support
- Object compression
- Improved insert/erase performance

## References

**Priority R-Tree**: A Practically Efficient and Worst-Case Optimal R-Tree
Lars Arge, Mark de Berg, Herman Haverkort, Ke Yi
SIGMOD 2004
[Paper](https://www.cse.ust.hk/~yike/prtree/)

## License

See LICENSE file for details.
