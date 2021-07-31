import unittest

import numpy as np

from qsdopt.qsd_optimizer import curvature


class QSDTests(unittest.TestCase):
    def test_curvature(self):
        def func(x, y):
            return (
                x ** 3 + x * y ** 2,
                np.array([3 * x ** 2 + y ** 2, 2 * x * y]),
                np.array([[6 * x, 2 * y], [2 * y, 2 * x]]),
            )
        f, g, H = func(0, 0.01)
        k = curvature(H, g, np.linalg.norm(g))
        self.assertEqual(k, 200.)

        f, g, H = func(0, 1.)
        k = curvature(H, g, np.linalg.norm(g))
        self.assertEqual(k, 2.)
