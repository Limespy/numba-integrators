from typing import NamedTuple

import numba as nb
import numba_integrators as ni
import numpy as np
from scipy.integrate import RK23
from scipy.integrate import RK45

scipy_integrators = {ni.RK23: RK23,
                     ni.RK45: RK45}
# ======================================================================
@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def riccati_differential(x, y):
    '''https://en.wikipedia.org/wiki/Bernoulli_differential_equation#Example
    '''
    return 2 * y/x - x ** 2 * y ** 2
# ----------------------------------------------------------------------
def riccati_solution(x):
    '''For initial value (1, 1)'''
    return x ** 2 / (x**5 / 5 + 4/5)
# ----------------------------------------------------------------------
class Riccati:
    differential = riccati_differential
    initial = (1., np.array((1.,), dtype = np.float64))
    solution = riccati_solution
    x_end = 10.
    y_end = riccati_solution(10.)
# ======================================================================
@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def sine_differential(x, y):
    '''sine wave'''
    return np.array((y[1], -y[0]))
# ----------------------------------------------------------------------
def sine_solution(x):
    '''For initial value (0, 1)'''
    return np.array((np.sin(x), np.cos(x)), dtype = np.float64)
# ----------------------------------------------------------------------
class Sine:
    differential = sine_differential
    initial = (0., np.array((0., 1.), dtype = np.float64))
    solution = sine_solution
    x_end = 10.
    y_end = sine_solution(10.)
# ======================================================================
def scipy_solve(solver_type, function, x0, y0, x_end, rtol, atol):

    solver = scipy_integrators[solver_type](function, x0, y0, x_end,
                                            atol = atol, rtol = rtol)
    while solver.status == "running":
        solver.step()
    assert solver.t == x_end
    return solver.y
