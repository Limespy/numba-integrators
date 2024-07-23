"""Implementation of the basic RK solvers as Numba Structref classes.

Mainly to test how much faster the solver object initilaisation would be
with these
"""
import numpy as np

from .._aux import calc_error
from .._aux import IterableNamespace
from .._aux import MAX_FACTOR
from .._aux import MIN_FACTOR
from .._aux import nbDecC
from .._aux import npAFloat64
from .._aux import RK23_params
from .._aux import RK45_params
from .._aux import SAFETY
from .._aux import step_prep
from ._first_aux import ODE1Type
from ._first_aux import Solver1
from ._structref_generated import RK
from .basic import select_initial_step
# ======================================================================
@nbDecC
def step(state: RK) -> bool:
    h_abs = state.x_bound - state.x
    if state.direction * h_abs <= 0: # x_bound has been reached
        return False
    x0 = state.x
    x = state.x
    y0 = state.y
    y = state.y
    h_abs = state.h_abs
    direction = state.direction

    min_step, h_abs = step_prep(x0, h_abs, direction, h_abs, state.max_step)

    K = state.K
    A = state.A
    C = state.C
    rtol = state.rtol
    atol = state.atol
    fun = state.fun
    K[0] = K[-1]
    y0_abs = np.abs(y0)
    _len = 1. / len(y0)
    while True: # := not working

        h = direction * h_abs

        # RK core loop

        Dx = C[0] * h
        K[1] = fun(x0 + Dx, y0 + Dx * K[0])

        for s in range(2, state.n_stages):
            K[s] = fun(x0 + C[s - 1] * h,
                       y0 + np.dot(A[s - 2,:s] * h, K[:s]))

        # Last step
        x = x0 + h
        y = y0 + np.dot(A[-2,:-1] * h, K[:-1])
        K[-1] = fun(x, y)

        state.nfev += state.n_stages

        error = calc_error(np.dot(A[-1], K), y, y0_abs, rtol, atol
                           ) * _len * h * h

        if error < 1:
            h_abs *= (MAX_FACTOR if error == 0 else
                    min(MAX_FACTOR, SAFETY * error ** state.error_exponent))
            if h_abs < min_step: # Due to the SAFETY, the step can shrink
                h_abs = min_step
            state.K = K
            state.h_abs = h_abs # type: ignore[misc]
            state.step_size = h # type: ignore[misc]
            state.x = x # type: ignore[misc]
            state.y = y # type: ignore[misc]
            return True # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR,
                                SAFETY * error ** state.error_exponent)
            if h_abs < min_step:
                state.K = K
                state.h_abs = h_abs # type: ignore[misc]
                state.step_size = h # type: ignore[misc]
                state.x = x # type: ignore[misc]
                state.y = y # type: ignore[misc]
                return False # Too small step size
# ======================================================================
@nbDecC
def init_RK(fun: ODE1Type,
            x0: np.float64,
            y0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params: tuple) -> RK:
    n_stages, A, C, error_exponent = solver_params
    K = np.zeros((n_stages + np.int64(1), len(y0)), dtype = y0.dtype)
    K[-1] = fun(x0, y0)
    direction = np.sign(x_bound - x0)

    if not first_step:
        h_abs = select_initial_step(
            fun, x0, y0, K[-1], direction,
            error_exponent, rtol, atol)
    else:
        h_abs = np.abs(first_step)
    step_size = direction * h_abs

    return RK(fun,
              x0,
              y0,
              rtol,
              atol,
              x_bound,
              max_step,
              h_abs,
              direction,
              step_size,
              np.int64(1),
              error_exponent,
              n_stages,
              A,
              C,
              K)
# ----------------------------------------------------------------------
class _Solver1_RK(Solver1):
    _init = init_RK
# ======================================================================
class RK23(_Solver1_RK):
    """Interface for creating RK45 solver state."""
    _solver_params = RK23_params
# ----------------------------------------------------------------------
class RK45(_Solver1_RK):
    """Interface for creating RK45 solver state."""
    _solver_params = RK45_params
# ----------------------------------------------------------------------
class Solvers(IterableNamespace):
    RK23 = RK23
    RK45 = RK45
