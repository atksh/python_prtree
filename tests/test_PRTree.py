import unittest
from python_prtree import PRTree
import numpy as np
import os
import time

class TestPRTree(unittest.TestCase):
    def test_result(self):
        def has_intersect(x, y):
            b1 = max(x[0], y[0]) <= min(x[1], y[1])
            b2 = max(x[2], y[2]) <= min(x[3], y[3])
            return b1 and b2

        idx = np.arange(100)
        x = np.random.rand(len(idx), 4).astype(np.float32)
        x[:, 1] += x[:, 0]
        x[:, 3] += x[:, 2]

        prtree = PRTree(idx, x)
        out = prtree.batch_query(x)
        for i in range(len(idx)):
            tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k])]
            self.assertEqual(set(out[i]), set(tmp))


        prtree.save('tree.bin')
        time.sleep(.3)
        prtree = PRTree("tree.bin")

        out = prtree.batch_query(x)
        for i in range(len(idx)):
            tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k])]
            self.assertEqual(set(out[i]), set(tmp))

        prtree = PRTree()
        prtree.load('tree.bin')

        out = prtree.batch_query(x)
        for i in range(len(idx)):
            tmp = [k for k in range(len(idx)) if has_intersect(x[i], x[k])]
            self.assertEqual(set(out[i]), set(tmp))


        os.remove('tree.bin')
        N= 100000
        idx = np.arange(N)
        x = np.random.rand(N, 4).astype(np.float32)
        x[:, 1] = x[:, 0] + x[:, 1] / np.sqrt(N) / 100
        x[:, 3] = x[:, 2] + x[:, 3] / np.sqrt(N) / 100
        prtree1 = PRTree(idx, x)

        prtree2 = PRTree(idx[:N//2], x[:N//2])
        for i in range(N//2, N):
            prtree2.insert(idx[i], x[i])

        x = np.random.rand(N, 4).astype(np.float32)
        x[:, 1] = x[:, 0] + x[:, 1] / np.sqrt(N) / 100
        x[:, 3] = x[:, 2] + x[:, 3] / np.sqrt(N) / 100
        for i in range(N):
            self.assertEqual(set(prtree1.query(x[i])), set(prtree2.query(x[i])))

        for i in range(N-1):
            prtree1.erase(i)
            prtree2.erase(i)

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(TestPRTree))
    return suite

