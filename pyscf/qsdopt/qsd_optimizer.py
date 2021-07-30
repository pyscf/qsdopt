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


def qsd_step(x0, g, H, sm3, stationary_point, step=1e-1):
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

    if inc_norm > step:
        # Search u value.
        umax = 1e0
        while np.abs(inc_norm - step) > step_thres:
            u = (umax + umin) / 2.0
            incNM = (u ** hNM - 1e0) * delta
            inc = NMeigvec.T @ incNM / sm3
            inc_norm = np.linalg.norm(inc)
            if inc_norm < step:
                umax = u
            else:
                umin = u
    print("u, max norm, real inc", umin, u, np.linalg.norm(delta), np.linalg.norm(inc))
    return inc


class QSD(lib.StreamObject):
    """QSD optimizer."""

    def __init__(self, method, stationary_point="min"):
        self.method = method
        self.stationary_point = stationary_point
        assert self.stationary_point in ["min", "TS"]

    def kernel(self, hess_update_freq):
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
        converged, self.mol = kernel(g_scanner, self.stationary_point)
