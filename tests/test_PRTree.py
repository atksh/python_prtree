import numpy as np
import pytest

from python_prtree import PRTree2D, PRTree3D, PRTree4D

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


def test_readme():
    idxes = np.array([1, 2])
    rects = np.array([[0.0, 0.0, 1.0, 0.5], [1.0, 1.5, 1.2, 3.0]])
    prtree = PRTree2D(idxes, rects)

    # batch query
    q = np.array([[0.5, 0.2, 0.6, 0.3], [0.8, 0.5, 1.5, 3.5]])
    result = prtree.batch_query(q)
    assert result == [[1], [1, 2]]

    # Insert
    prtree.insert(3, [1.0, 1.0, 2.0, 2.0])
    q = np.array([[0.5, 0.2, 0.6, 0.3], [0.8, 0.5, 1.5, 3.5]])
    result = prtree.batch_query(q)
    assert result == [[1], [1, 2, 3]]

    # Erase
    prtree.erase(2)
    result = prtree.batch_query(q)
    assert result == [[1], [1, 3]]

    # non-batch query
    assert prtree.query([0.5, 0.5, 1.0, 1.0]) == [1, 3]

    # point query
    assert prtree.query([0.5, 0.5]) == [1]
    assert prtree.query(0.5, 0.5) == [1]


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_disjoint_small_gap(PRTree, dim):
    """Test for the bug reported by Matteo Lacki where boxes with small gaps are incorrectly reported as intersecting.
    
    This was caused by float32 precision loss where values like 75.02750896 and 75.02751435
    would collapse to the same float32 value (75.02751159667968750000).
    """
    if dim == 2:
        A = np.array([[72.47410062, 80.52848893, 75.02750896, 85.40646976]])
        B = np.array([[75.02751435, 80.0, 78.71358218, 85.0]])
        gap_dim = 0
    elif dim == 3:
        A = np.array([[72.47410062, 80.52848893, 54.68197159, 75.02750896, 85.40646976, 62.42859506]])
        B = np.array([[75.02751435, 74.65699325, 61.09751679, 78.71358218, 82.4585436, 67.24904609]])
        gap_dim = 0
    else:  # dim == 4
        A = np.array([[72.47410062, 80.52848893, 54.68197159, 60.0, 75.02750896, 85.40646976, 62.42859506, 70.0]])
        B = np.array([[75.02751435, 74.65699325, 61.09751679, 55.0, 78.71358218, 82.4585436, 67.24904609, 65.0]])
        gap_dim = 0
    
    assert A[0][gap_dim + dim] < B[0][gap_dim], f"A_max ({A[0][gap_dim + dim]}) should be < B_min ({B[0][gap_dim]})"
    gap = B[0][gap_dim] - A[0][gap_dim + dim]
    assert gap > 0, f"Gap should be positive, got {gap}"
    
    tree = PRTree(np.array([0]), A)
    
    result = tree.batch_query(B)
    assert result == [[]], f"Expected [[]] (no intersection), got {result}. Gap was {gap}"
    
    result_query = tree.query(B[0])
    assert result_query == [], f"Expected [] (no intersection), got {result_query}. Gap was {gap}"


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_touching_boxes(PRTree, dim):
    """Test that boxes that exactly touch (share a boundary) are considered intersecting.
    
    This documents the intended closed-interval semantics where touching counts as intersecting.
    """
    A = np.zeros((1, 2 * dim))
    B = np.zeros((1, 2 * dim))
    
    for i in range(dim):
        A[0][i] = 0.0  # min coords
        A[0][i + dim] = 1.0  # max coords
        B[0][i] = 1.0  # min coords
        B[0][i + dim] = 2.0  # max coords
    
    tree = PRTree(np.array([0]), A)
    
    result = tree.batch_query(B)
    assert result == [[0]], f"Expected [[0]] (touching boxes intersect), got {result}"
    
    result_query = tree.query(B[0])
    assert result_query == [0], f"Expected [0] (touching boxes intersect), got {result_query}"


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_large_magnitude_coords(PRTree, dim):
    """Test boxes with large magnitude coordinates to ensure precision is maintained."""
    A = np.zeros((1, 2 * dim))
    B = np.zeros((1, 2 * dim))
    
    base = 1e6
    for i in range(dim):
        A[0][i] = base + i  # min coords
        A[0][i + dim] = base + i + 1.0  # max coords
        B[0][i] = base + i + 1.1  # min coords (larger gap for double precision limits)
        B[0][i + dim] = base + i + 2.0  # max coords
    
    tree = PRTree(np.array([0]), A)
    
    result = tree.batch_query(B)
    assert result == [[]], f"Expected [[]] (no intersection at large magnitude), got {result}"


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_degenerate_boxes(PRTree, dim):
    """Test boxes with zero volume (degenerate in one or more dimensions)."""
    A = np.zeros((1, 2 * dim))
    B = np.zeros((1, 2 * dim))
    
    for i in range(dim):
        if i == 0:
            A[0][i] = 1.0
            A[0][i + dim] = 1.0
            B[0][i] = 1.0
            B[0][i + dim] = 1.0
        else:
            A[0][i] = 0.0
            A[0][i + dim] = 1.0
            B[0][i] = 0.5
            B[0][i + dim] = 1.5
    
    tree = PRTree(np.array([0]), A)
    
    result = tree.batch_query(B)
    assert result == [[0]], f"Expected [[0]] (degenerate boxes intersect), got {result}"


@pytest.mark.parametrize("PRTree, dim", [(PRTree2D, 2), (PRTree3D, 3), (PRTree4D, 4)])
def test_query_vs_batch_query_consistency(PRTree, dim):
    """Test that query and batch_query return consistent results."""
    np.random.seed(42)
    N = 50
    idx = np.arange(N)
    x = np.random.rand(N, 2 * dim) * 100  # Use larger range to test precision
    for i in range(dim):
        x[:, i + dim] += x[:, i] + 0.1  # Ensure valid boxes with small width
    
    tree = PRTree(idx, x)
    
    queries = np.random.rand(20, 2 * dim) * 100
    for i in range(dim):
        queries[:, i + dim] += queries[:, i] + 0.1
    
    batch_results = tree.batch_query(queries)
    for i, query in enumerate(queries):
        single_result = tree.query(query)
        assert set(batch_results[i]) == set(single_result), \
            f"Query {i}: batch_query returned {batch_results[i]}, query returned {single_result}"


def test_save_load_float64_matteo_case(tmp_path):
    """Regression test: ensure idx2exact survives save/load for float64 input.
    
    This tests the fix for the serialization bug where idx2exact was not being
    archived, causing trees built from float64 input to lose correctness after
    save/load. The Matteo bug case has boxes separated by ~5.4e-6, which requires
    double-precision refinement to correctly identify as disjoint.
    """
    import gc
    
    A = np.array([[72.47410062, 80.52848893, 54.68197159, 75.02750896, 85.40646976, 62.42859506]], dtype=np.float64)
    B = np.array([[75.02751435, 74.65699325, 61.09751679, 78.71358218, 82.4585436, 67.24904609]], dtype=np.float64)
    
    assert A[0][3] < B[0][0], "Test setup error: boxes should be disjoint"
    gap = B[0][0] - A[0][3]
    assert 5e-6 < gap < 6e-6, f"Test setup error: expected gap ~5.4e-6, got {gap}"
    
    tree = PRTree3D(np.array([0], dtype=np.int64), A)
    
    result_before = tree.batch_query(B)
    assert result_before == [[]], f"Before save: Expected [[]] (disjoint), got {result_before}"
    
    fname = tmp_path / "tree_float64.bin"
    fname = str(fname)
    tree.save(fname)
    
    del tree
    gc.collect()
    
    tree_loaded = PRTree3D(fname)
    
    result_after = tree_loaded.batch_query(B)
    assert result_after == [[]], f"After load: Expected [[]] (disjoint), got {result_after}"
    
    np.random.seed(42)
    queries = np.random.rand(10, 6).astype(np.float64) * 100
    for i in range(3):
        queries[:, i + 3] += queries[:, i] + 1e-5  # Small gaps
    
    results_before_save = tree_loaded.batch_query(queries)
    
    fname2 = tmp_path / "tree_float64_2.bin"
    fname2 = str(fname2)
    tree_loaded.save(fname2)
    del tree_loaded
    gc.collect()
    
    tree_loaded2 = PRTree3D(fname2)
    results_after_save = tree_loaded2.batch_query(queries)
    
    assert results_before_save == results_after_save, \
        "Random queries: results changed after save/load cycle"


def test_save_load_float32_no_regression(tmp_path):
    """Regression test: ensure float32 path still works correctly after save/load.
    
    This tests that the serialization fix (adding idx2exact to archive) doesn't
    break the float32 path, which doesn't use idx2exact.
    """
    import gc
    
    np.random.seed(42)
    N = 100
    idx = np.arange(N, dtype=np.int64)
    x = np.random.rand(N, 6).astype(np.float32) * 100
    for i in range(3):
        x[:, i + 3] += x[:, i] + 1.0  # Ensure valid boxes
    
    tree = PRTree3D(idx, x)
    
    queries = np.random.rand(20, 6).astype(np.float32) * 100
    for i in range(3):
        queries[:, i + 3] += queries[:, i] + 1.0
    
    results_before = tree.batch_query(queries)
    
    fname = tmp_path / "tree_float32.bin"
    fname = str(fname)
    tree.save(fname)
    del tree
    gc.collect()
    
    tree_loaded = PRTree3D(fname)
    results_after = tree_loaded.batch_query(queries)
    
    assert results_before == results_after, \
        "Float32 path: results changed after save/load cycle"
