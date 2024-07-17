import numba as nb
import numpy as np

from ._aux import Arrayable
from ._aux import convert
from ._aux import IS_CACHE
from ._aux import MAX_FACTOR
from ._aux import MIN_FACTOR
from ._aux import nbODEtype
from ._aux import norm
from ._aux import npAFloat64
from ._aux import ODEFUN
from ._aux import RK45_params
from ._aux import SAFETY
from ._basic import select_initial_step
from ._structref_generated import RK
# from ._basic import _step

@nb.njit(cache = IS_CACHE)
def step(state: RK) -> bool:
    if state.direction * (state.x - state.x_bound) >= 0: # t_bound has been reached
        return False
    x_old = state.x
    x = state.x
    y_old = state.y
    y = state.y
    h_abs = state.h_abs
    eps = np.abs(np.nextafter(x_old, state.direction * np.inf) - x_old)
    min_step = 8 * eps

    if h_abs < min_step:
        h_abs = min_step

    while True: # := not working
        if h_abs > state.max_step:
            h_abs = state.max_step - eps
        h = h_abs * state.direction
        # Updating
        x = x_old + h
        state.K[0] = state.K[-1] # type: ignore[misc]

        if state.direction * (x - state.x_bound) >= 0: # End reached
            x = state.x_bound
            h = x - x_old
            h_abs = np.abs(h) # There is something weird going on here
        # RK core loop
        for s in range(1, state.n_stages):
            state.K[s] = state.fun(x_old + state.C[s] * h,
                       y_old + np.dot(state.K[:s].T, state.A[s,:s]) * h)

        y = y_old + h * np.dot(state.K[:-1].T, state.B)

        state.K[-1] = state.fun(x, y) # type: ignore[misc]

        error_norm = norm(np.dot(state.K.T, state.E)
                          * h
                          / (state.atol + np.maximum(np.abs(y_old),
                                              np.abs(y)) * state.rtol))

        if error_norm < 1:
            h_abs *= (MAX_FACTOR if error_norm == 0 else
                            min(MAX_FACTOR,
                                SAFETY * error_norm ** state.error_exponent))
            state.h_abs = h_abs # type: ignore[misc]
            state.step_size = h # type: ignore[misc]
            state.x = x # type: ignore[misc]
            state.y = y # type: ignore[misc]
            return True # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR,
                                SAFETY * error_norm ** state.error_exponent)
            if h_abs < min_step:
                state.h_abs = h_abs # type: ignore[misc]
                state.step_size = h # type: ignore[misc]
                state.x = x # type: ignore[misc]
                state.y = y # type: ignore[misc]
                return False # Too small step size

@nb.njit(cache = IS_CACHE)
def _init_RK(fun: ODEFUN,
            x0: np.float64,
            y0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            error_estimator_order: np.int8,
            n_stages: np.int8,
            A: npAFloat64,
            B: npAFloat64,
            C: npAFloat64,
            E: npAFloat64):

    K = np.zeros((n_stages + 1, len(y0)), dtype = y0.dtype)
    K[-1] = fun(x0, y0)
    direction = np.float64(1. if x_bound == x0 else np.sign(x_bound - x0))
    error_exponent = -1 / (error_estimator_order + 1)

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
              error_exponent,
              n_stages,
              A,
              B,
              C,
              E,
              K)
# ----------------------------------------------------------------------
def RK45(fun: ODEFUN,
         x0: float,
         y0: Arrayable,
         x_bound: float,
         max_step: float = np.inf,
         rtol: Arrayable = 1e-3,
         atol: Arrayable = 1e-6,
         first_step: float = 0.) -> RK:
    """Interface for creating RK45 solver state."""
    y0, rtol, atol = convert(y0, rtol, atol)
    return _init_RK(fun,
                    np.float64(x0),
                    y0,
                    np.float64(x_bound),
                    np.float64(max_step),
                    rtol,
                    atol,
                    np.float64(first_step),
                    *RK45_params)
