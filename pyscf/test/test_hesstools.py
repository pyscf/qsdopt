import unittest

import numpy as np
from numpy.testing import assert_allclose

from pyscf import gto
from qsdopt.hesstools import numhess


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
