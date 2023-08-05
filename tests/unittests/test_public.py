'''Unittests for public interface of the package.
Classes are sorted alphabetically and related functions'''
import numba as nb
import numba_integrators as ni
import numpy as np
import pytest
from numba_integrators._aux import Float64Array

# ======================================================================
# Auxiliaries

@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def function_to_integrate(t: float, y: Float64Array):
    '''Function that integrates to sine and cosine'''
    return np.array((y[1], -y[0]))

y0 = np.array((0., 1.), dtype = np.float64)
# ======================================================================
class Test_RK45:
    def test_initialise(self):
        x_end = 1
        solver = ni.RK45(function_to_integrate,
                        0.,
                        y0,
                        x_end,
                        rtol = 1e-8,
                        atol = 1e-8)
        while solver.step():
            pass
        assert solver.t == x_end
        assert np.abs(solver.y[0] / np.sin(x_end) -1) < 1e-7
