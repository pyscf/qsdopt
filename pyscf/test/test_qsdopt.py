import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from pyscf import gto, scf
from qsdopt.qsd_optimizer import QSD


class QSDOptTests(unittest.TestCase):
    def test_H2O_TS(self):
        mol = gto.M(atom="O 0 0 0; H 0 0 1.2; H 0, 0.1, -1.2", basis="minao", verbose=0)
        mf = scf.RHF(mol)
        optimizer = QSD(mf, stationary_point="TS")
        optimizer.kernel()
        TS = np.array(
            [
                [1.50228475e-18, 1.04977710e-02, 3.08561792e-07],
                [1.51292813e-17, -6.81250364e-02, 1.88752430e00],
                [-1.21073939e-17, 8.91333136e-02, -1.88752923e00],
            ]
        )
        assert_almost_equal(TS, mol.atom_coords())
        print(TS - mol.atom_coords())
