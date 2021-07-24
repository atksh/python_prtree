# python_prtree

*python_prtree* is a python/c++ implementation of the Priority R-Tree (see references below). The supported futures are as follows:

- Construct a Priority R-Tree (PRTree) from an array of rectangles
  - The array shape is (xmin, ymin, xmax, ymax) in 2D and (xmin, ymin, zmin, xmax, ymax, zmax) in 3D.
  - 3D PRTree is supported since `>=0.4.0`.
  - Changed the ordering of the array shape since `>=0.5.0`. 
- `query` and `batch_query` with rectangle(s)
  - Supports Point query with (x, y) in 2D and (x, y, z) in 3D since `>=0.5.1`
- `insert` and `erase` (but not yet optimized)
  - Fixed a bug that one cannot insert to an empty PRTree at `0.5.0`.
- `rebuild` with already given data since `>=0.5.0`.
  - For better performance when too many insert/erase operations are called since.
- The `insert` and `query` methods can now be passed pickable Python objects instead of int64 indexes since `>=0.5.2`.
  - See the example below for more details.

This package is mainly for **mostly static situations** where insertion and deletion events rarely occur (e.g. map matching).

## Installation
You can install python_prtree with the pip command:
```bash
pip install python-prtree
```

If the pip installation does not work (e.g. on an M1 Mac), please git clone clone and install as follows:
```bash
pip install -U cmake pybind11
git clone --recursive https://github.com/atksh/python_prtree
cd python_prtree
python setup.py install
```

## A Simple Example
```python
import numpy as np
from python_prtree import PRTree2D

idxes = np.array([1, 2])  # must be unique because it uses idx as key for hash map
rects = np.array([[0.0, 0.0, 1.0, 0.5],
                  [1.0, 1.5, 1.2, 3.0]])  # (xmin, ymin, xmax, ymax)

prtree = PRTree2D(idxes, rects)  # initial construction

q = np.array([[0.5, 0.2, 0.6, 0.3],
              [0.8, 0.5, 1.5, 3.5]])
result = prtree.batch_query(q)
print(result)
# [[1], [1, 2]]

# You can insert an additional rectangle,
prtree.insert(3, np.array([1.0, 1.0, 2.0, 2.0]))
q = np.array([[0.5, 0.2, 0.6, 0.3],
              [0.8, 0.5, 1.5, 3.5]])
result = prtree.batch_query(q)
print(result)
# [[1], [1, 2, 3]]

# And erase by index.
prtree.erase(2)
result = prtree.batch_query(q)
print(result)
# [[1], [1, 3]]

# Point query
print(prtree.query(0.5, 0.5))
# [1]
print(prtree.query((0.5, 0.5)))
# [1]
```

## New Features and Changes 
### `python-prtree>=0.5.3`
- Add gzip compression for pickled objects.

### `python-prtree>=0.5.2`
You can use pickable Python objects instead of int64 indexes for `insert` and `query` methods:

```python
import numpy as np
from python_prtree import PRTree2D

objs = [{"name": "foo"}, (1, 2, 3)]  # must NOT be unique but pickable
rects = np.array([[0.0, 0.0, 1.0, 0.5],
                  [1.0, 1.5, 1.2, 3.0]])  # (xmin, ymin, xmax, ymax)

prtree = PRTree2D()
for obj, rect in zip(objs, rects):
    # keyword argments: bb(bounding box) and obj(object)
    prtree.insert(bb=rect, obj=obj)

# returns indexes genereted by incremental rule.
result = prtree.query((0, 0, 1, 1))
print(result)
# [1]

# returns objects when you specify the keyword argment return_obj=True
result = prtree.query((0, 0, 1, 1), return_obj=True)
print(result)
# [{'name': 'foo'}]
```

### `python-prtree>=0.5.0`
- [**CRUTIAL**] Changed the input order from (xmin, xmax, ymin, ymax, ...) to (xmin, ymin, xmax, ymax, ...).
- [**FEATURE**] Added rebuild method to build the PRTree from scratch using the already given data.
- [**BUGFIX**] Fixed a bug that prevented insertion into an empty PRTree.
- [**REMIND**] Cross-version saving and loading compatibility is not guaranteed.

### `python-prtree>=0.4.0`
You can use PRTree3D:

```python
import numpy as np
from python_prtree import PRTree3D

idxes = np.array([1, 2])  # must be unique because it uses idx as key for hash map
rects = np.array([[0.0, 0.5, 0.0, 0.5, 1.0, 0.5],
                  [1.0, 1.5, 2.0, 1.2, 2.5, 3.0]])  # (xmin, ymin, zmin, xmax, ymax, zmax)

prtree = PRTree3D(idxes, rects)  # initial construction

q = np.array([[0.5, 0.2, 0.2, 0.6, 0.3, 0.3],
              [0.8, 0.5, 0.5, 1.5, 3.5, 3.5]])
result = prtree.batch_query(q)
print(result)
# [[], [2]]
```

### `python-prtree>=0.3.0`
You can save and load a binary file as follows:

```python
# save
prtree.save('tree.bin')


# load with binary file
prtree = PRTree('tree.bin')

# or defered load
prtree = PRTree()
prtree.load('tree.bin')
```

Cross-version compatibility is **NOT** guaranteed, so please reconstruct your tree when you update this package.

## Note

The 1d-array batch query will be implicitly treated as a batch with size = 1.
If you want 1d result, please use `query` method.
```python
result = prtree.query(q[0])
print(result)
# [1]

result = prtree.batch_query(q[0])
print(result)
# [[1]]
```


You can also erase(delete) by index and insert a new one.
```python
prtree.erase(1)  # delete the rectangle with idx=1 from the PRTree

prtree.insert(3, np.array([0.3, 0.5, 0.1, 0.2]))  # add a new rectangle to the PRTree
```




# Performance
## Construction
### 2d

![2d_fig1](https://raw.githubusercontent.com/atksh/python_prtree/master/docs/images/2d_fig1.png)

### 3d

![3d_fig1](https://raw.githubusercontent.com/atksh/python_prtree/master/docs/images/3d_fig1.png)

## Query and batch query

### 2d

![2d_fig2](https://raw.githubusercontent.com/atksh/python_prtree/master/docs/images/2d_fig2.png)

### 3d

![3d_fig2](https://raw.githubusercontent.com/atksh/python_prtree/master/docs/images/3d_fig2.png)

## Delete and insert

### 2d

![2d_fig3](https://raw.githubusercontent.com/atksh/python_prtree/master/docs/images/2d_fig3.png)

### 3d

![3d_fig3](https://raw.githubusercontent.com/atksh/python_prtree/master/docs/images/3d_fig3.png)

# Requirement
- numpy


# NOTE

- C++ implements this PRTree with Pybind11 and much faster than the numba implementation of PRTree.
- If you can use C++, you should use boost::geometry (I did not know it and sadly made this package).
- Please note that insert / erase operations are not optimized compared to ordinary r-tree. Plus, this implementation does not exactly follow that of the paper due to my technical skills.


# Reference
The Priority R-Tree: A Practically Efficient and Worst-Case Optimal R-Tree
Lars Arge, Mark de Berg, Herman Haverkort, and Ke Yi
Proceedings of the 2004 ACM SIGMOD International Conference on Management of Data (SIGMOD '04), Paris, France, June 2004, 347-358. Journal version in ACM Transactions on Algorithms.
[author's page](https://www.cse.ust.hk/~yike/prtree/)
