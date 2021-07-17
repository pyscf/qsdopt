import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyscf import gto, scf
from qsdopt.hesstools import filter_hessian, numhess

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
