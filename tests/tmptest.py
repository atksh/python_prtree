import gc
import random
from python_prtree import PRTree
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18


def f(N):
    idx = np.arange(N)
    x = np.random.rand(N, 4)
    x[:, 1] = x[:, 0] + x[:, 1] / np.sqrt(N) / 100
    x[:, 3] = x[:, 2] + x[:, 3] / np.sqrt(N) / 100
    s = time.time()
    prtree = PRTree(idx, x)
    t = time.time()
    x = np.random.rand(100_000, 4)
    x[:, 1] = x[:, 0] + x[:, 1] / np.sqrt(N) / 100
    x[:, 3] = x[:, 2] + x[:, 3] / np.sqrt(N) / 100
    t2 = time.time()
    out = prtree.batch_query(x)
    u = time.time()
    out = [prtree.query(y) for y in x]
    u1 = time.time()
    del_list = random.sample(idx.tolist(), k=min(N, 100_000))
    u2 = time.time()
    for k in del_list:
        prtree.erase(k)
    v = time.time()

    x = np.random.rand(min(N, 100_000), 4)
    x[:, 1] = x[:, 0] + x[:, 1] / np.sqrt(N) / 100
    x[:, 3] = x[:, 2] + x[:, 3] / np.sqrt(N) / 100
    v2 = time.time()
    for i, k in enumerate(del_list):
        prtree.insert(k, x[i])
    return t - s, u - t2, (u1 - u), (v - u2), (time.time() - v2)


x = []
y1 = []
y2 = []
y22 = []
y3 = []
y4 = []
for i in list(range(7, 25))[::-1]:
    n = int(1.5 ** (12 + i))
    print(n, end=', ')
    x.append(n)
    s, t, t2, u, v = f(n)
    y1.append(s)
    y2.append(t)
    y22.append(t2)
    y3.append(u)
    y4.append(v)
    gc.collect()
