'''Unittests for public interface of the package.
Classes are sorted alphabetically and related functions'''
from itertools import product
from typing import Callable

import numba as nb
import numba_integrators as ni
import numpy as np
import pytest
from numba_integrators import reference as ref
from numba_integrators._aux import npAFloat64
# ======================================================================
# Auxiliaries
def compare_to_scipy(solver_type, rtol, atol, problem):
    solver = solver_type(problem.differential,
                         *problem.initial,
                         problem.x_end,
                         rtol = rtol, atol = atol)

    while ni.step(solver):
        pass
    assert solver.t == problem.x_end

    y_ni = solver.y
    err_ni = np.sum(np.abs(y_ni - problem.y_end))

    y_scipy = ref.scipy_solve(solver_type, # type: ignore
                              problem.differential,
                              *problem.initial,
                              problem.x_end,
                              rtol, atol)
    err_scipy = np.sum(np.abs(y_scipy - problem.y_end))
    assert err_ni < err_scipy*1.01
# ======================================================================
class Test_Basic:
    '''Testing agains scipy relative to analytical'''
    @pytest.mark.parametrize('solver_type, problem',
                             product((ni.RK23, ni.RK45),
                                     (ref.Riccati, ref.Sine)))
    def test_to_scipy(self, solver_type, problem):
        compare_to_scipy(solver_type, 1e-10, 1e-10, problem)
# ======================================================================
@nb.njit()
def f_advanced(t, y, p):
    dy = np.array((y[1], -y[0]))
    a = (p[0] * t, p[1]*t)
    return dy, a

sine_diff = ref.Sine.differential

@nb.njit()
def f_sine_advanced(t, y, p):
    return sine_diff(t, y), p
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
        solver = Solver(f_advanced, *ref.Sine.initial, self.parameters, ref.Sine.x_end)
        assert isinstance(solver.auxiliary, tuple)
        assert solver.auxiliary[0] == self.parameters[0] * solver.t
        assert np.all(solver.auxiliary[1] == self.parameters[1] * solver.t)
        ni.step(solver)
        assert solver.auxiliary[0] == self.parameters[0] * solver.t
        assert np.all(solver.auxiliary[1] == self.parameters[1] * solver.t)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('solver_type', (ni.RK23, ni.RK45))
    def test_identical_to_base(self, solver_type):
        solver_base = solver_type(ref.Sine.differential,
                                  *ref.Sine.initial,
                                  ref.Sine.x_end)
        Solver = ni.Advanced(self.parameters_type, self.parameters_type, solver_type)
        solver_advanced = Solver(f_sine_advanced,
                                 *ref.Sine.initial,
                                 self.parameters,
                                 ref.Sine.x_end)
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
