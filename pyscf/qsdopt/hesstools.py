# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np


def numhess(mol, g_scanner):
    """Evaluate numerical hessian of the energy."""
    delta = 1e-4
    fourdelta = 4 * delta
    geom = mol.atom_coords()
    nat = geom.shape[0]
    ndim = 3 * nat
    H = np.zeros([ndim, ndim])

    for iat in range(nat):
        for icoor in range(3):
            i = 3 * iat + icoor
            _geom = geom.copy()
            _geom[iat, icoor] = geom[iat, icoor] + delta
            mol.set_geom_(_geom, unit="Bohr")
            e1, g1 = g_scanner(mol)

            _geom[iat, icoor] = geom[iat, icoor] - delta
            mol.set_geom_(_geom, unit="Bohr")
            e2, g2 = g_scanner(mol)

            H[i, :] = (g1 - g2).reshape(-1)

    H = (H + H.T) / fourdelta
    return H
