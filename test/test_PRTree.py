import unittest
from python_prtree import PRTree
import numpy as np

class TestPRTree(unittest.TestCase):
    def test_result(self):
        def has_intersect(x, y):
            b1 = max(x[0], y[0]) <= min(x[1], y[1])
            b2 = max(x[2], y[2]) <= min(x[3], y[3])
            return b1 and b2

        idx = np.arange(100)
        x = np.random.rand(len(idx), 4)
        x[:, 1] += x[:, 0]
        x[:, 3] += x[:, 2]

        prtree = PRTree(idx, x)
        out = prtree.find_all(x)
        for i in range(len(idx)):
            tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k])]
            self.assertEqual(set(out[i]), set(tmp))


def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestPRTree))
    return suite

