# python_prtree

"python_prtree" is an implementation of Priority R-Tree (see reference below).
Supported futures are as follows:

- Construct Priority R-Tree(PRTree) from rectangles
- query and batch query with rectangle(s)
- insert and delete (but not optimized yet)


# Usage 
```python
import numpy as np
from python_prtree import PRTree

idxes = np.array([1, 2])
rects = np.array([[0.0, 1.0, 0.0, 0.5],
                  [1.0, 1.2, 2.5, 3.0]])  # (xmin, xmax, ymin, ymax)

prtree = PRTree(idxes, rects)

q = np.array([[0.5, 0.6, 0.2, 0.3],
              [0.8, 1.5, 0.5, 3.5]])
result = prtree.batch_query(q)
print(result)
# [[1], [1, 2]]
```

1d-array query will be implicitly treated as batch with size = 1.
If you want 1d result, please use `query` method.
```python
result = prtree.query(q[0])
print(result)
# [1]

result = prtree.batch_query(q[0])
print(result)
# [[1]]
```

# Installation
Install python_prtree with pip command.
```bash
pip install git@github.com:atksh/python_prtree.git
```

Or, you can clone and pip just like
```bash
git clone --recursive git@github.com:atksh/python_prtree.git
cd python_prtree
python setup.py test; rm -rf build  #(optional)
pip install .
```

# Performance
## Construction
![fig1](https://github.com/atksh/python_prtree/docs/images/fig1.png)

## Query and batch query
![fig2](https://github.com/atksh/python_prtree/docs/images/fig2.png)

## Delete and insert
![fig3](https://github.com/atksh/python_prtree/docs/images/fig3.png)

# Requirement
- numpy


# NOTE

- This PRTree is implemented by C++ with Pybind11, and much faster than numba implementation of PRTree.
- Methods exposed to python are `init` and `find_all`. No other methods are avaiable from python.
- If you can use C++, you should use boost::geometry (I didn't know it and sadly made this package).


# Reference
The Priority R-Tree: A Practically Efficient and Worst-Case Optimal R-Tree
Lars Arge, Mark de Berg, Herman Haverkort, and Ke Yi
Proceedings of the 2004 ACM SIGMOD International Conference on Management of Data (SIGMOD '04), Paris, France, June 2004, 347-358. Journal version in ACM Transactions on Algorithms.
[author's page](https://www.cse.ust.hk/~yike/prtree/)
