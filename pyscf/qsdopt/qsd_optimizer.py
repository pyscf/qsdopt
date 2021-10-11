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

from pyscf import lib
from pyscf.grad.rhf import GradientsMixin
from pyscf.qsdopt.hesstools import (
    filter_hessian,
    hess_BFGS_update,
    hess_powell_update,
    numhess,
)


def kernel(
    g_scanner,
    stationary_point,
    hess_update_rule,
    hess_update_freq=0,
    numhess_method="forward",
    max_iter=100,
    step=0.1,
    hmin=1e-6,
    gthres=1e-5,
):
    """
    Quadratic Steepest Descent method for transition state and minima optimization.
    Implemeted following: J. Chem. Phys., 101, 2157, 1994; 10.1063/1.467721
    Arguments:
        hess_update_freq: Iterations before next numerical hessian calculation. If 0 the
            hessian is not updated in the optimization process unless problematic region is reached.
        numhess:method: Method for numerical hessian calculation. "forward" for forward differences
            calculation, "central" for central differences calculation.
        max_iter: Maximum number of iterations in the optimization process.
        step: Maximum step at each optimization iteration.
        hmin: Minimum distance between next point (x1) and quadratic form minima (m) to make x1 = m.
        gthres: Gradient threshold to consider stationary point reached.

    """
    converged = False
    ITA = 0
    ITAM = 10
    m = g_scanner.mol.atom_mass_list()
    sm = np.sqrt(m)
    sm3 = np.repeat(sm, 3)
    nat = len(m)
    x0 = g_scanner.mol.atom_coords().flatten()
    energy, g0 = g_scanner(g_scanner.mol)
    g0 = g0.flatten()
    H = numhess(g_scanner.mol, g_scanner, g0, numhess_method)
    H = filter_hessian(g_scanner.mol, H)
    H = np.einsum("ij, i, j -> ij", H, 1 / sm3, 1 / sm3)
    inc = _qsd_step(x0 * sm3, g0 / sm3, H, sm3, stationary_point, step=step)
    x1 = x0 + inc

    print("Iteration   Energy       Gradient Norm   Increment norm")
    for it in range(1, max_iter):
        x_1 = x0.copy()
        g_1 = g0.copy()
        x0 = x1.copy()
        g_scanner.mol.set_geom_(x0.reshape(nat, 3), unit="Bohr")
        energy, g0 = g_scanner(g_scanner.mol)
        g0 = g0.flatten()
        gnorm = np.linalg.norm(g0)
        incnorm = np.linalg.norm(inc)
        print(f"{it:3d}         {energy:10.7f}  {gnorm:4.2e}        {incnorm:4.2e}")
        if gnorm < gthres or incnorm < hmin or ITA > ITAM:
            converged = True
            break
        if hess_update_freq > 0 and it % hess_update_freq == 0 or ITA >= 4:
            print("Recalculating hessian")
            H = numhess(g_scanner.mol, g_scanner, g0, numhess_method)
        else:
            dH = hess_update_rule(H, x0 - x_1, g0 - g_1)
            H += dH
        Hf = filter_hessian(g_scanner.mol, H)
        Hf = np.einsum("ij, i, j -> ij", Hf, 1 / sm3, 1 / sm3)
        inc = _qsd_step(x0 * sm3, g0 / sm3, Hf, sm3, stationary_point, step=step)
        x1 = x0 + inc

        val = (x_1 - x0) @ (x0 - x1)
        if val > 0.0:
            ITA = 0
        else:
            ITA += 1

    g_scanner.mol.set_geom_(x1.reshape(nat, 3), unit="Bohr")
    return converged, g_scanner.mol


def _qsd_step(x0, g, H, sm3, stationary_point, step=1e-1):
    zero_thres = 1e-10
    step_thres = 1e-1 * step
    eigval, eigvec = np.linalg.eigh(H)
    NMeigvec = eigvec.T[np.abs(eigval) > zero_thres]
    hNM = eigval[np.abs(eigval) > zero_thres]
    gNM = NMeigvec @ g
    delta = gNM / hNM
    if stationary_point == "TS":
        hNM[0] = -hNM[0]

    inc = NMeigvec.T @ delta / sm3
    if np.linalg.norm(inc) < step:
        return -inc

    if all(hNM > 0):
        umin = 0e0
    else:
        umin = 1e-25

    incNM = (umin ** hNM - 1e0) * delta
    inc = NMeigvec.T @ incNM / sm3
    inc_norm = np.linalg.norm(inc)
    inc_max = inc_norm

    if inc_norm > step:
        # Search u value.
        umax = 1e0
        inc_min = 0e0
        while (
            np.abs(inc_norm - step) > step_thres and np.abs(inc_max - inc_min) > 1e-20
        ):
            u = (umax + umin) / 2.0
            incNM = (u ** hNM - 1e0) * delta
            inc = NMeigvec.T @ incNM / sm3
            inc_norm = np.linalg.norm(inc)
            if inc_norm < step:
                umax = u
                inc_min = inc_norm
            else:
                umin = u
                inc_max = inc_norm
    return inc


class QSD(lib.StreamObject):
    """QSD optimizer."""

    def __init__(self, method, stationary_point="min"):
        assert stationary_point in ["min", "TS"]
        self.method = method
        self.stationary_point = stationary_point
        self.converged = False
        if self.stationary_point == "TS":
            self.hess_update = hess_powell_update
        elif self.stationary_point == "min":
            self.hess_update = hess_BFGS_update

    def kernel(self, **kwargs):
        if isinstance(self.method, lib.GradScanner):
            g_scanner = self.method
        elif isinstance(self.method, GradientsMixin):
            g_scanner = self.method.as_scanner()
        elif getattr(self.method, "nuc_grad_method", None):
            g_scanner = self.method.nuc_grad_method().as_scanner()
        else:
            raise NotImplementedError(
                "Nuclear gradients of %s not available" % self.method
            )
        self.converged, self.mol = kernel(
            g_scanner, self.stationary_point, self.hess_update, **kwargs
        )


if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.M(atom="O 0 0 0; H 0 0 1.2; H 0, 0.1, -1.2", basis="minao", verbose=0)
    mf = scf.RHF(mol)
    optimizer = QSD(mf, stationary_point="TS")
    optimizer.kernel()
    print(f"Converged: {optimizer.converged}")
    print(mol.atom_coords())
