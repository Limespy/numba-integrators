'''Unittests for public interface of the package.
Classes are sorted alphabetically and related functions'''
import numba as nb
import numba_integrators as ni
import numpy as np
import pytest
from numba_integrators import reference as ref
from numba_integrators._aux import npAFloat64
# ======================================================================
pytestmark = pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaExperimentalFeatureWarning")
# ======================================================================
# Auxiliaries
def compare_to_scipy(solver_type, rtol, atol, problem):
    solver = solver_type(problem.differential,
                         problem.t0,
                         problem.y0,
                         problem.x_end,
                         rtol = rtol, atol = atol)

    while ni.step(solver):
        pass
    assert solver.t == problem.x_end

    y_ni = solver.y
    err_ni = np.sum(np.abs(y_ni - problem.y_end))

    y_scipy = ref.scipy_solve(solver_type, # type: ignore
                              problem.differential,
                              problem.t0,
                              problem.y0,
                              problem.x_end,
                              rtol, atol)
    err_scipy = np.sum(np.abs(y_scipy - problem.y_end))
    assert err_ni < err_scipy*1.01
# ======================================================================
class Test_Basic:
    '''Testing agains scipy relative to analytical'''
    @pytest.mark.parametrize('solver_type', ni.ALL)
    def test_to_scipy(self, solver_type):
        compare_to_scipy(solver_type, 1e-10, 1e-10, ref.riccati)
# ======================================================================
@nb.njit()
def f_advanced(t: float, y: npAFloat64, p):
    dy = np.array((y[1], -y[0]))
    a = (p[0] * t, p[1]*t)
    return dy, a
# ----------------------------------------------------------------------
sine_diff = ref.sine.differential
@nb.njit()
def f_sine_advanced(t: float, y: npAFloat64, p):
    return sine_diff(t, y), p
# ----------------------------------------------------------------------
class Test_Advanced:

    parameters_type = nb.types.Tuple((nb.float64, nb.float64[:]))
    auxiliary_type = nb.types.Tuple((nb.float64, nb.float64[:]))
    parameters = (1., np.array((1., 1.), dtype = np.float64))
    # ------------------------------------------------------------------
    def test_all_class_creation(self):
        Solvers = ni.Advanced(self.parameters_type, self.auxiliary_type, ni.Solver.ALL)
        assert isinstance(Solvers, dict)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('solver_type', (ni.RK23, ni.RK45))
    def test_identical_to_base(self, solver_type):
        solver_base = solver_type(ref.sine.differential,
                                  ref.sine.t0,
                                  ref.sine.y0,
                                  ref.sine.x_end)
        Solver = ni.Advanced(self.parameters_type, self.parameters_type, solver_type)
        assert Solver is not None
        assert not isinstance(Solver, dict)
        solver_advanced = Solver(f_sine_advanced,
                                 ref.sine.t0,
                                 ref.sine.y0,
                                 self.parameters,
                                 ref.sine.x_end)
        assert isinstance(solver_advanced.auxiliary, tuple)
        assert solver_advanced.auxiliary[0] == self.parameters[0]
        assert np.all(solver_advanced.auxiliary[1] == self.parameters[1])

        assert solver_base.t == solver_advanced.t
        assert np.all(solver_base.y == solver_advanced.y)
        assert solver_advanced.auxiliary == self.parameters

        ni.step(solver_base)
        ni.step(solver_advanced)

        assert solver_base.t == solver_advanced.t
        assert np.all(solver_base.y == solver_advanced.y)

        assert isinstance(solver_advanced.auxiliary, tuple)
        assert solver_advanced.auxiliary[0] == self.parameters[0]
        assert np.all(solver_advanced.auxiliary[1] == self.parameters[1])

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
# ======================================================================
@nb.njit()
def condition_sine(solver, parameters: float) -> bool:
    return solver.y[0] > parameters
# ----------------------------------------------------------------------
class Test_fast_forward:
    def test_t(self):
        solver = ni.RK45(ref.sine.differential,
                         ref.sine.t0,
                         ref.sine.y0,
                         ref.sine.x_end)
        t_range = ref.sine.x_end - ref.sine.t0
        t_step1 = t_range / 10.1
        ni.ff2t(solver, t_step1)
        assert solver.t == t_step1
        assert solver.t_bound == ref.sine.x_end

        while ni.ff2t(solver, solver.t + t_step1):
            ...
        assert solver.t == solver.t_bound
        assert solver.t_bound == ref.sine.x_end
    # ------------------------------------------------------------------
    def test_condition(self):
        solver = ni.RK45(ref.sine.differential,
                         ref.sine.t0,
                         ref.sine.y0,
                         ref.sine.x_end)
        condition_parameter = 0.5
        assert ni.ff2cond(solver, condition_sine, condition_parameter)
        assert condition_sine(solver, condition_parameter)
