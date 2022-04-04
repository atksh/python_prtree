import pytest
from python_prtree import PRTree2D, PRTree3D, PRTree4D
import numpy as np


N_SEED = 5


def has_intersect(x, y, dim):
    return all([max(x[i], y[i]) <= min(x[i + dim], y[i + dim]) for i in range(dim)])


@pytest.mark.parametrize("seed", range(N_SEED))
@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_result(seed, PRTree, dim):
    np.random.seed(seed)
    idx = np.arange(100)
    x = np.random.rand(len(idx), 2 * dim)
    for i in range(dim):
        x[:, i + dim] += x[:, i]

    prtree = PRTree(idx, x)
    out = prtree.batch_query(x)
    for i in range(len(idx)):
        tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k], dim)]
        assert set(out[i]) == set(tmp)

    out = [prtree.query(x[i]) for i in range(len(x))]
    for i in range(len(idx)):
        tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k], dim)]
        assert set(out[i]) == set(tmp)

    # test point query
    x[:, dim:] = x[:, :dim]
    out1 = prtree.batch_query(x)
    out2 = prtree.batch_query(x[:, :dim])
    for i in range(len(idx)):
        assert set(out1[i]) == set(out2[i])


@pytest.mark.parametrize("seed", range(N_SEED))
@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_io(seed, PRTree, dim, tmp_path):
    np.random.seed(seed)
    idx = np.arange(100)
    x = np.random.rand(len(idx), 2 * dim)
    for i in range(dim):
        x[:, i + dim] += x[:, i]

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


@pytest.mark.parametrize("seed", range(N_SEED))
@pytest.mark.parametrize("from_scratch", [False, True])
@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_insert_erase(seed, from_scratch, rebuild, PRTree, dim):
    np.random.seed(seed)
    N = 10000
    idx = np.arange(N)
    x = np.random.rand(N, 2 * dim)
    for i in range(dim):
        x[:, i + dim] = x[:, i] + x[:, i + dim] / 10
    prtree1 = PRTree(idx, x)
    if rebuild:
        prtree1.rebuild()

    if from_scratch:
        prtree2 = PRTree()
        for i in range(N):
            assert prtree2.size() == i
            prtree2.insert(idx[i], x[i])
    else:
        prtree2 = PRTree(idx[: N // 2], x[: N // 2])
        for i in range(N // 2, N):
            assert prtree2.size() == i
            prtree2.insert(idx[i], x[i])

    x = np.random.rand(100, 2 * dim)
    for i in range(dim):
        x[:, i + dim] = x[:, i] + x[:, i + dim] / 10
    for i in range(x.shape[0]):
        assert set(prtree1.query(x[i])) == set(prtree2.query(x[i]))

    for i in range(N // 2):
        prtree1.erase(i)
        prtree2.erase(i)

    for i in range(dim):
        x[:, i + dim] = x[:, i] + x[:, i + dim] / 10
    for i in range(x.shape[0]):
        assert set(prtree1.query(x[i])) == set(prtree2.query(x[i]))


@pytest.mark.parametrize("seed", range(N_SEED))
@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_obj(seed, PRTree, dim, tmp_path):
    np.random.seed(seed)
    x = np.random.rand(100, 2 * dim)
    for i in range(dim):
        x[:, i + dim] += x[:, i]

    obj = [(i, (i, str(i))) for i in range(len(x))]
    prtree = PRTree()
    prtree2 = PRTree()
    for i in range(len(x)):
        prtree.insert(i, x[i])
        prtree2.insert(bb=x[i], obj=obj[i])

    q = (0,) * dim + (1,) * dim
    idx = prtree.query(q)
    return_obj = prtree2.query(q, return_obj=True)
    assert len(idx) > 0
    assert set(return_obj) == set([obj[i] for i in idx])

    fname = tmp_path / "tree.bin"
    fname = str(fname)
    prtree.save(fname)
    prtree = PRTree(fname)

    idx = prtree.query(q)
    return_obj = prtree2.query(q, return_obj=True)
    assert set(return_obj) == set([obj[i] for i in idx])
