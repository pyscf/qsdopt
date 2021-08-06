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

from pyscf.hessian.thermo import _get_TR


def numhess(mol, g_scanner, g0, method):
    if method == "forward":
        H = forward_differences_hess(mol, g_scanner, g0)
    elif method == "central":
        H = central_differences_hess(mol, g_scanner)
    return H


def forward_differences_hess(mol, g_scanner, g0):
    """Evaluate numerical hessian of the energy using forward differences."""
    delta = 1e-4
    twodelta = 2 * delta
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

            H[i, :] = g1.reshape(-1) - g0

    H = (H + H.T) / twodelta
    return H


def central_differences_hess(mol, g_scanner):
    """Evaluate numerical hessian of the energy using central differences."""
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


def filter_hessian(mol, H):
    """Projects translations and rotations out of the hessian."""
    # NOTE: This method can maybe be in pyscf.hessian.thermo
    zero_thres = 1e-5
    coords = mol.atom_coords()
    m = mol.atom_mass_list()
    sm = np.sqrt(m)
    sm3 = np.repeat(sm, 3)
    nat = len(m)

    D = np.zeros([6, 3 * nat])
    D[0], D[1], D[2], D[3], D[4], D[5] = _get_TR(m, coords)
    norm = np.linalg.norm(D, axis=1)
    D = D[norm > zero_thres]
    D /= norm[norm > zero_thres][:, None]

    P = np.identity(D.shape[1]) - D.T @ D
    H = np.einsum("ij, i, j -> ij", H, 1 / sm3, 1 / sm3)
    H = P.T @ H @ P
    H = np.einsum("ij, i, j -> ij", H, sm3, sm3)
    return H


def hess_powell_update(H, dq, dg):
    """Update hessian using Powell rule"""
    Hdq = H.dot(dq)
    dqdq = dq.dot(dq)
    aux = (dg - Hdq) * dq[:, None]
    dH1 = (aux + aux.T) / dqdq
    dH2 = (dq.dot(dg) - dq.dot(Hdq)) * dq[:, None] * dq / dqdq ** 2
    dH = dH1 - dH2
    return dH


def hess_BFGS_update(H, dq, dg):
    """Update hessian using BFGS rule"""
    dH1 = dg[None, :] * dg[:, None] / dq.dot(dg)
    dH2 = H.dot(dq[None, :] * dq[:, None]).dot(H) / dq.dot(H).dot(dq)
    dH = dH1 - dH2  # BFGS update
    return dH


if __name__ == "__main__":
    from pyscf import gto, scf

    test_names = ["H2O", "H2"]
    atoms = [
        "O 0. 0.40846003 0.60306451; H 0. -0.770609 2.20504746; H 0. 2.2518751 1.3492855",
        "H 0 0 0; H 0 0 1.603",
    ]

    for (name, atom) in zip(test_names, atoms):
        mol = gto.M(
            atom=atom,
            basis="minao",
            verbose=0,
            unit="Bohr",
        )
        mf = scf.RHF(mol)
        g_scanner = mf.nuc_grad_method().as_scanner()
        H = numhess(mol, g_scanner)
        eigval = np.linalg.eigvalsh(H)

        H = filter_hessian(mol, H)
        feigval = np.linalg.eigvalsh(H)

        print(f"\n {name} example")
        print("Not filtered; Filtered")
        for ival in range(eigval.shape[0]):
            print(f"{eigval[ival]:2.5E} {feigval[ival]:2.5E}")
