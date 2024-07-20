import numba as nb
import numba_integrators as ni
import numpy as np
from numba_integrators.first import basic as b1
from numba_integrators.first import structref as sr
# ======================================================================
@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def _ODE(t: float, y):
    return np.array((y[1], -y[0]))
# ----------------------------------------------------------------------
y0 = np.array((0., 1.))

args = (_ODE, 0.0, y0)
kwargs = dict(x_bound = 20000 * np.pi,
                atol = 1e-8,
                rtol = 1e-8)
# ======================================================================
def RK45():
    solver = b1.RK45(*args, **kwargs)
    while ni.step(solver):
        ...
# ======================================================================
def RK45_structref():
    solver = sr.RK45(*args, **kwargs)
    while sr.step(solver):
        ...
