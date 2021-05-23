import pytest
from python_prtree import PRTree2D, PRTree3D
import numpy as np


def has_intersect(x, y, dim):
    return all(
        [max(x[2 * i], y[2 * i]) <= min(x[2 * i + 1], y[2 * i + 1]) for i in range(dim)]
    )


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
def test_result(PRTree, dim):
    idx = np.arange(100)
    x = np.random.rand(len(idx), 2 * dim)
    for i in range(dim):
        x[:, 2 * i + 1] += x[:, 2 * i]

    prtree = PRTree(idx, x)
    out = prtree.batch_query(x)
    for i in range(len(idx)):
        tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k], dim)]
        assert set(out[i]) == set(tmp)


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
def test_io(PRTree, dim, tmp_path):
    idx = np.arange(100)
    x = np.random.rand(len(idx), 2 * dim)
    for i in range(dim):
        x[:, 2 * i + 1] += x[:, 2 * i]

    prtree = PRTree(idx, x)

    fname = tmp_path / "tree.bin"
    fname = str(fname)
    prtree.save(fname)
    prtree = PRTree(fname)

    out = prtree.batch_query(x)
    for i in range(len(idx)):
        tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k], dim)]
        assert set(out[i]) == set(tmp)

    prtree = PRTree()
    prtree.load(fname)

    out = prtree.batch_query(x)
    for i in range(len(idx)):
        tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k], dim)]
        assert set(out[i]) == set(tmp)


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3)])
def test_insert_erase(PRTree, dim):
    N = 10000
    idx = np.arange(N)
    x = np.random.rand(N, 2 * dim)
    for i in range(dim):
        x[:, 2 * i + 1] = x[:, 2 * i] + x[:, 2 * i + 1] / np.sqrt(N) / 100
    prtree1 = PRTree(idx, x)

    prtree2 = PRTree(idx[: N // 2], x[: N // 2])
    for i in range(N // 2, N):
        prtree2.insert(idx[i], x[i])

    x = np.random.rand(100, 2 * dim)
    for i in range(dim):
        x[:, 2 * i + 1] = x[:, 2 * i] + x[:, 2 * i + 1] / np.sqrt(N) / 100
    for i in range(x.shape[0]):
        assert set(prtree1.query(x[i])) == set(prtree2.query(x[i]))

    for i in range(N // 2):
        prtree1.erase(i)
        prtree2.erase(i)

    for i in range(dim):
        x[:, 2 * i + 1] = x[:, 2 * i] + x[:, 2 * i + 1] / np.sqrt(N) / 100
    for i in range(x.shape[0]):
        assert set(prtree1.query(x[i])) == set(prtree2.query(x[i]))
