from python_prtree import PRTree2D
import numpy as np

def f(N, PRTree, dim):
    idx = np.arange(N)
    x = np.random.rand(N, 2*dim).astype(np.float32)
    print(x.nbytes // 1024 // 1024) # mb
    for i in range(dim):
        x[:, i+dim] = x[:, i] + x[:, i+dim] / np.sqrt(N) / 100
    prtree = PRTree(idx, x)
    x = np.random.rand(100_000, 2*dim).astype(np.float32)
    for i in range(dim):
        x[:, i+dim] = x[:, i] + x[:, i+dim] / np.sqrt(N) / 100
    prtree.batch_query(x)

if __name__ == "__main__":
    f(10_000_000, PRTree2D, dim=2)
