from PRTree import PRTree
import numpy as np

idx = np.range(100)
x = np.random.randn(len(idx), 4)
x[:, 1] += x[:, 0]
x[:, 3] += x[:, 2]

prtree = PRTree(idx, x)
out = prtree.find_all(x)
for i in range(len(idx)):
    assert i in out[i]
