import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyscf import gto, scf
from qsdopt.hesstools import (
    filter_hessian,
    hess_BFGS_update,
    hess_powell_update,
    numhess,
)

zero_thres = 1e-15


class MyScanner:
    def __call__(self, mol):
        x = mol.atom_coords()
        return x[0, 0] ** 2 * x[0, 1] + x[0, 2] ** 2, np.array(
            [2 * x[0, 0] * x[0, 1], x[0, 0] ** 2, 2 * x[0, 2]]
        )


class HessianTests(unittest.TestCase):
    def test_numhess(self):
        scanner = MyScanner()
        mol = gto.M(atom="O 1 2 3", basis="minao", unit="Bohr")
        H = numhess(mol, scanner)
        assert_allclose(
            H, np.array([[4.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        )

    def test_filterhess1(self):
        mol = gto.M(
            atom="O 0. 0.40846003 0.60306451; H 0. -0.770609 2.20504746; H 0. 2.2518751 1.3492855",
            basis="minao",
            verbose=0,
            unit="Bohr",
        )
        mf = scf.RHF(mol)
        g_scanner = mf.nuc_grad_method().as_scanner()
        H = numhess(mol, g_scanner)

        H = filter_hessian(mol, H)
        feigval = np.linalg.eigvalsh(H)
        self.assertEqual(len(feigval[feigval < zero_thres]), 6)

    def test_filterhess2(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 1.603",
            basis="minao",
            verbose=0,
            unit="Bohr",
        )
        mf = scf.RHF(mol)
        g_scanner = mf.nuc_grad_method().as_scanner()
        H = numhess(mol, g_scanner)

        H = filter_hessian(mol, H)
        feigval = np.linalg.eigvalsh(H)
        self.assertEqual(len(feigval[feigval < zero_thres]), 5)

    def test_hess_powell_update(self):
        def func(x, y):
            return (
                x ** 3 + x * y ** 2,
                np.array([3 * x ** 2 + y ** 2, 2 * x * y]),
                np.array([[6 * x, 2 * y], [2 * y, 2 * x]]),
            )

        x1 = np.array([0.1, 0.2])
        x2 = np.array([0.1, 0.3])

        f1, g1, H1 = func(x1[0], x1[1])
        f2, g2, H2 = func(x2[0], x2[1])

        dH = hess_powell_update(H1, (x2 - x1), (g2 - g1))
        assert_allclose(
            dH,
            np.array([[0.0, 1e-1], [1e-1, 0.0]]),
            atol=1e-15
        )

    def test_hess_BFGS_update(self):
        def func(x, y):
            return (
                x ** 3 + x * y ** 2,
                np.array([3 * x ** 2 + y ** 2, 2 * x * y]),
                np.array([[6 * x, 2 * y], [2 * y, 2 * x]]),
            )

        x1 = np.array([0.1, 0.2])
        x2 = np.array([0.1, 0.3])

        f1, g1, H1 = func(x1[0], x1[1])
        f2, g2, H2 = func(x2[0], x2[1])

        dH = hess_BFGS_update(H1, (x2 - x1), (g2 - g1))
        assert_allclose(
            dH,
            np.array([[4.5e-1, 1e-1], [1e-1, 0.0]]),
            atol=1e-15
        )
