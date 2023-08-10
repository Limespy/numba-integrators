'''Unittests for public interface of the package.
Classes are sorted alphabetically and related functions'''
from typing import Callable

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
t0 = 0.
y0 = np.array((0., 1.), dtype = np.float64)
# ======================================================================
class Test_RK45:
    def test_initialise(self):
        x_end = 1
        solver = ni.RK45(function_to_integrate,
                        t0,
                        y0,
                        x_end,
                        rtol = 1e-8,
                        atol = 1e-8)
        while solver.step():
            pass
        assert solver.t == x_end
        assert np.abs(solver.y[0] / np.sin(x_end) -1) < 1e-7
# ======================================================================

@nb.njit()
def f_advanced(t, y, p):
    dy = np.array((y[1], -y[0]))
    a = (p[0] * t, p[1]*t)
    return dy, a
# ----------------------------------------------------------------------
class Test_Advanced:
    parameters_type = nb.types.Tuple((nb.float64, nb.float64[:]))
    auxiliary_type = nb.types.Tuple((nb.float64, nb.float64[:]))
    parameters = (1., np.array((1., 1.), dtype = np.float64))
    @pytest.mark.parametrize('solver_type', (ni.RK23, ni.RK45))
    def test_class_creation(self, solver_type):
        Solver = ni.Advanced(nb.float64, nb.float64, solver_type)
        assert Solver is not None
        assert not isinstance(Solver, dict)
    # ------------------------------------------------------------------
    def test_all_class_creation(self):
        Solvers = ni.Advanced(self.parameters_type, self.auxiliary_type, ni.Solver.ALL)
        assert isinstance(Solvers, dict)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('solver_type', (ni.RK23, ni.RK45))
    def test_class_use(self, solver_type):
        Solver = ni.Advanced(self.parameters_type, self.auxiliary_type, solver_type)
        solver = Solver(f_advanced, t0, y0, self.parameters, 1.)
        assert isinstance(solver.auxiliary, tuple)
        assert solver.auxiliary[0] == self.parameters[0] * solver.t
        assert np.all(solver.auxiliary[1] == self.parameters[1] * solver.t)
        ni.step(solver)
        assert solver.auxiliary[0] == self.parameters[0] * solver.t
        assert np.all(solver.auxiliary[1] == self.parameters[1] * solver.t)
    # ------------------------------------------------------------------
