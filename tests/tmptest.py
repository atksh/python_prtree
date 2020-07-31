import gc
import random
from python_prtree import PRTree
import numpy as np
import time
import gc


def f(N):
    idx = np.arange(N)
    x = np.random.rand(N, 4)
    x[:, 1] = x[:, 0] + x[:, 1] / np.sqrt(N) / 100
    x[:, 3] = x[:, 2] + x[:, 3] / np.sqrt(N) / 100
    s = time.time()
    print(x.nbytes // 1024 // 1024, 'mb')
    prtree = PRTree(idx, x)
    #prtree.save(f'{N}.bin')
    del prtree
    gc.collect()
    return time.time() - s


for _ in range(1000):
    f(10_000_000)
