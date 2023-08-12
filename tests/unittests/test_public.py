'''Unittests for public interface of the package.
Classes are sorted alphabetically and related functions'''
from typing import Callable

import numba as nb
import numba_integrators as ni
import numpy as np
import pytest
from numba_integrators import reference as ref
from numba_integrators._aux import npAFloat64
# ======================================================================
# Auxiliaries

@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def function_to_integrate(t: float, y: npAFloat64):
    '''Function that integrates to sine and cosine'''
    return np.array((y[1], -y[0]))
t0 = 0.
y0 = np.array((0., 1.), dtype = np.float64)
x_end = 1.
# ======================================================================
class Test_Basic:
    @pytest.mark.parametrize('solver_type', (ni.RK23, ni.RK45))
    def test_sine(self, solver_type):
        tol = 1e-8
        y_analytical = np.sin(x_end)
        solver = solver_type(function_to_integrate,
                        t0,
                        y0,
                        x_end,
                        rtol = tol,
                        atol = tol)
        while ni.step(solver):
            pass
        assert solver.t == x_end

        limit = np.abs(y_analytical)*10*tol
        assert np.abs(solver.y[0] - y_analytical) < limit
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('solver_type', (ni.RK23, ni.RK45))
    def test_time_dependent(self, solver_type):
        tol = 1e-8
        x_end = 10.
        y_analytical = ref.riccati_solution(x_end)
        solver = solver_type(ref.riccati_differential, *ref.riccati_initial,
                        x_end,
                        rtol = tol,
                        atol = tol)
        while ni.step(solver):
            pass
        assert solver.t == x_end

        limit = np.abs(y_analytical)*10*tol
        assert np.abs(solver.y[0] - y_analytical) < limit
# ======================================================================
@nb.njit()
def f_advanced(t, y, p):
    dy = np.array((y[1], -y[0]))
    a = (p[0] * t, p[1]*t)
    return dy, a

@nb.njit()
def f_sine_advanced(t, y, p):
    return function_to_integrate(t, y), p
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
        solver = Solver(f_advanced, t0, y0, self.parameters, x_end)
        assert isinstance(solver.auxiliary, tuple)
        assert solver.auxiliary[0] == self.parameters[0] * solver.t
        assert np.all(solver.auxiliary[1] == self.parameters[1] * solver.t)
        ni.step(solver)
        assert solver.auxiliary[0] == self.parameters[0] * solver.t
        assert np.all(solver.auxiliary[1] == self.parameters[1] * solver.t)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('solver_type', (ni.RK23, ni.RK45))
    def test_identical_to_base(self, solver_type):
        solver_base = solver_type(function_to_integrate,
                                  t0,
                                  y0,
                                  x_end)
        Solver = ni.Advanced(self.parameters_type, self.parameters_type, solver_type)
        solver_advanced = Solver(f_sine_advanced,
                                 t0,
                                 y0,
                                 self.parameters,
                                 x_end)
        assert solver_base.t == solver_advanced.t
        assert np.all(solver_base.y == solver_advanced.y)
        assert solver_advanced.auxiliary == self.parameters
        ni.step(solver_base)
        ni.step(solver_advanced)
        assert solver_base.t == solver_advanced.t
        assert np.all(solver_base.y == solver_advanced.y)

        x_base = []
        y_base = []
        x_advanced = []
        y_advanced = []

        while ni.step(solver_base):
            assert ni.step(solver_advanced)
            x_base.append(solver_base.t)
            y_base.append(solver_base.y)
            x_advanced.append(solver_advanced.t)
            y_advanced.append(solver_advanced.y)

        assert x_base == x_advanced
        assert all(np.all(y_b == y_a) for y_b, y_a in zip(y_base, y_advanced))
    # ------------------------------------------------------------------
