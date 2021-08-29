import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from pyscf import gto, scf
from pyscf.qsdopt.qsd_optimizer import QSD


class QSDOptTests(unittest.TestCase):
    def test_H2O_TS(self):
        mol = gto.M(atom="O 0 0 0; H 0 0 1.2; H 0, 0.1, -1.2", basis="minao", verbose=0)
        mf = scf.RHF(mol)
        optimizer = QSD(mf, stationary_point="TS")
        optimizer.kernel()
        coords = mol.atom_coords()
        v1 = coords[0, :] - coords[1, :]
        v2 = coords[0, :] - coords[2, :]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        phi = np.arccos(v1.dot(v2) / (d1 * d2))
        assert_almost_equal(d1, d2, 4)
        assert_almost_equal(phi, np.pi, 4)
        self.assertTrue(optimizer.converged)

    def test_H2O_min(self):
        mol = gto.M(atom="O 0 0 0; H 0 0 1.2; H 0, 0.5, -1.2", basis="minao", verbose=0)
        mf = scf.RHF(mol)
        optimizer = QSD(mf, stationary_point="min")
        optimizer.kernel()
        coords = mol.atom_coords()
        v1 = coords[0, :] - coords[1, :]
        v2 = coords[0, :] - coords[2, :]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        phi = np.arccos(v1.dot(v2) / (d1 * d2))
        assert_almost_equal(d1, d2, 4)
        assert_almost_equal(phi, np.radians(104.3), 4)
        self.assertTrue(optimizer.converged)
