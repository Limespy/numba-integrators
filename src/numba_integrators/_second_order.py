"""Basic RK integrators implemented with numba jitclass."""
from typing import Any

import numba as nb
import numpy as np

from ._aux import Arrayable
from ._aux import calc_error_norm
from ._aux import calc_tolerance
from ._aux import convert
from ._aux import h_prep
from ._aux import IS_CACHE
from ._aux import IterableNamespace
from ._aux import MAX_FACTOR
from ._aux import MIN_FACTOR
from ._aux import nbA
from ._aux import nbARO
from ._aux import nbODE2_type
from ._aux import norm
from ._aux import npAFloat64
from ._aux import ODE2Type
from ._aux import RK23_params
from ._aux import RK45_params
from ._aux import RK_params_type
from ._aux import SAFETY
from ._aux import step_prep
from ._first_order import base_spec
# ======================================================================
@nb.njit(fastmath = True, cache = IS_CACHE)
def calc_h0(y0: npAFloat64,
            dy0: npAFloat64,
            ddy0: npAFloat64,
            direction: np.float64,
            scale: npAFloat64):
    d0 = norm(dy0 / scale)
    d1 = norm(ddy0 / scale)

    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    Dx = h0 * direction
    Ddy = Dx * ddy0
    dy1 = dy0 + Ddy
    y1 = y0 + dy0 * Dx + Ddy * 0.5 * Dx
    return h0, y1, dy1, d1
# ----------------------------------------------------------------------
@nb.njit(fastmath = True, cache = IS_CACHE)
def calc_h_abs(y_diff: npAFloat64,
               h0: np.float64,
               scale: npAFloat64,
               error_exponent: np.float64,
               d1: np.float64):
    d2 = norm(y_diff / scale) / h0

    return min(100. * h0,
               (max(1e-6, h0 * 1e-3) if d1 <= 1e-15 and d2 <= 1e-15
                else (max(d1, d2) * 100.) ** error_exponent))
# ----------------------------------------------------------------------
@nb.njit(nb.float64(nbODE2_type,
                    nb.float64,
                    nbA(1),
                    nbA(1),
                    nbA(1),
                    nb.float64,
                    nb.float64,
                    nbARO(1),
                    nbARO(1)),
         fastmath = True, cache = IS_CACHE)
def select_initial_step(fun: ODE2Type,
                        x0: np.float64,
                        y0: npAFloat64,
                        dy0: npAFloat64,
                        ddy0: npAFloat64,
                        direction: np.float64,
                        error_exponent: np.float64,
                        rtol: npAFloat64,
                        atol: npAFloat64) -> np.float64:
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    x0 : np.float64
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    dy0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(x0, y0)``.
    direction : np.float64
        Integration direction.
    error_exponent : np.float64
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : np.float64
        Desired relative tolerance.
    atol : np.float64
        Desired absolute tolerance.

    Returns
    -------
    h_abs : np.float64
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    scale = calc_tolerance(np.abs(dy0), rtol, atol)
    h0, y1, dy1, d1 = calc_h0(y0, dy0, ddy0, direction, scale)
    ddy1 = fun(x0 + h0 * direction, y1, dy1)
    return calc_h_abs(ddy1 - ddy0, h0, scale, error_exponent, d1)
# ----------------------------------------------------------------------
# @nb.njit(cache = IS_CACHE)
# def substeps1(fun, x_old, y_old, dy_old, h, A, B, C, K, K2, n_stages):
#     for s in range(1, n_stages):
#         ah = A[s,:s] * h
#         dy = dy_old + np.dot(ah, K2[:s])
#         K[s] = dy
#         K2[s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[:s]), dy)
#     Bh = B * h

#     Ddy = np.dot(Bh, K2[:-1])
#     dy = dy_old + Ddy
#     y = y_old + np.dot(Bh, K[:-1])# + adjustment

#     K[-1] = dy
#     K2[-1] = fun(x_old + h, y, dy)
#     return y, dy
# # ----------------------------------------------------------------------
# @nb.njit(cache = IS_CACHE)
# def substeps2(fun, x_old, y_old, dy_old, h, A, B, C, K, K2, n_stages):
#     _A = A.copy()
#     for i in range(1, len(C)):
#         _A[i] /= C[i]
#     for s in range(1, n_stages):
#         Dx = C[s] * h
#         dy = dy_old + np.dot(_A[s,:s] * Dx, K2[:s])
#         K[s] = dy
#         # scaler = C[s] / C[s+1]
#         Dy1 = np.dot(_A[s,:s] * Dx, K[:s])
#         # k = K[:s+1]
#         # a = _A[s+1,s+1]
#         Dy2 = np.dot(_A[s+1, :s] * Dx, K[:s])
#         y = y_old + 0.5 * (Dy1 + Dy2)
#         K2[s] = fun(x_old + Dx, y, dy)
#     # s = end
#     # Dx = C[s] * h
#     # dy = dy_old + np.dot(_A[s,:s] * Dx, K2[:s])
#     # K[s] = dy
#     # # scaler = C[s] / C[s+1]
#     # K2[s] = fun(x_old + Dx, y_old + np.dot(_A[s,:s] * Dx, K[:s]), dy)

#     Bh = B * h

#     Ddy = np.dot(Bh, K2[:-1])
#     dy = dy_old + Ddy
#     y = y_old + np.dot(Bh, K[:-1])# + adjustment

#     K[-1] = dy
#     K2[-1] = fun(x_old + h, y, dy)
#     return y, dy
# ----------------------------------------------------------------------
@nb.njit(cache = IS_CACHE)
def substeps3(fun, x_old, y_old, dy_old, h, A, B, C, K, K2, n_stages):

    Dx = C[1] * h

    dy = dy_old + K2[0] * Dx

    a = 0.5 * K2[0]
    b = K[0] - K2[0] * x_old
    c = y_old + x_old * (a * x_old - dy_old)

    x = x_old + Dx
    y = x * (a * x + b) + c
    K[1] = dy
    K2[1] = fun(x, y, dy)

    for s in range(2, n_stages):
        ah = A[s,:s] * h
        dy = dy_old + np.dot(ah, K2[:s])
        K[s] = dy
        K2[s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[:s]), dy)
    Bh = B * h

    Ddy = np.dot(Bh, K2[:-1])
    dy = dy_old + Ddy
    y = y_old + np.dot(Bh, K[:-1])# + adjustment

    K[-1] = dy
    K2[-1] = fun(x_old + h, y, dy)
    return y, dy
# ----------------------------------------------------------------------
@nb.njit(cache = IS_CACHE)
def step(fun: ODE2Type,
          direction: np.float64,
          x: np.float64,
          y: npAFloat64,
          dy: npAFloat64,
          x_bound: np.float64,
          h_abs: np.float64,
          max_step: np.float64,
          K: npAFloat64,
          K2: npAFloat64,
          n_stages: np.int8,
          rtol: npAFloat64,
          atol: npAFloat64,
          A: npAFloat64,
          B: npAFloat64,
          C: npAFloat64,
          E: npAFloat64,
          error_exponent: np.float64) -> tuple[bool,
                                          np.float64,
                                          npAFloat64,
                                          np.float64,
                                          np.float64,
                                          npAFloat64]:
    if direction * (x - x_bound) >= 0: # x_bound has been reached
        return False, x, y, dy, h_abs, direction *h_abs

    h_abs, x_old, eps, min_step = step_prep(h_abs, x, direction)
    y_old = y
    dy_old = dy
    K[0] = K[-1]
    K2[0] = K2[-1]
    while True: # := not working
        _, h, h_abs = h_prep(h_abs, max_step, eps, x_old, x_bound, direction)

        # _RK2 core loop
        # len(K) == n_stages + 1
        # y, dy = substeps3(fun, x_old, y_old, dy_old, h, A, B, C, K, K2, n_stages)

        # First step with 2nd degree polynomial
        Dx = C[1] * h

        dy = dy_old + K2[0] * Dx

        a = 0.5 * K2[0]
        b = K[0] - K2[0] * x_old
        c = y_old + x_old * (a * x_old - dy_old)

        x = x_old + Dx
        y = x * (a * x + b) + c
        K[1] = dy
        K2[1] = fun(x, y, dy)

        for s in range(2, n_stages):
            ah = A[s,:s] * h
            dy = dy_old + np.dot(ah, K2[:s])
            K[s] = dy
            K2[s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[:s]), dy)
        Bh = B * h

        Ddy = np.dot(Bh, K2[:-1])
        dy = dy_old + Ddy
        y = y_old + np.dot(Bh, K[:-1])# + adjustment

        x = x_old + h

        K[-1] = dy
        K2[-1] = fun(x, y, dy)
        # Error calculation
        error_norm = calc_error_norm(K, E, h, y, y_old, rtol, atol)

        if error_norm < 1.:
            h_abs *= (MAX_FACTOR if error_norm == 0. else
                      min(MAX_FACTOR, SAFETY * error_norm ** error_exponent))
            return True, x, y, dy, h_abs, h # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** error_exponent)
            if h_abs < min_step:
                return False, x, y, dy, h_abs, h # Too small step size
# ----------------------------------------------------------------------
@nb.experimental.jitclass(base_spec + (('dy', nbA(1)),
                                       ('fun', nbODE2_type),
                                       ('K2', nbA(2))))
class _RK2:
    """Base class for explicit Runge-Kutta methods."""

    def __init__(self,
                 fun: ODE2Type,
                 x0: np.float64,
                 y0: npAFloat64,
                 dy0: npAFloat64,
                 x_bound: np.float64,
                 max_step: np.float64,
                 rtol: npAFloat64,
                 atol: npAFloat64,
                 first_step: np.float64,
                 error_exponent: np.float64,
                 n_stages: np.int8,
                 A: npAFloat64,
                 B: npAFloat64,
                 C: npAFloat64,
                 E: npAFloat64):
        self.n_stages = n_stages
        self.A = A
        self.B = B
        self.C = C
        self.E = E
        self.fun = fun
        self.x = x0
        self.y = y0
        self.dy = dy0
        self.x_bound = x_bound
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step

        self.K = np.zeros((self.n_stages + 1, len(y0)), dtype = y0.dtype)
        self.K2 = self.K.copy()
        self.K[-1] = dy0
        self.K2[-1] = self.fun(self.x, self.y, self.dy)
        self.direction = 1. if x_bound == x0 else np.sign(x_bound - x0)
        self.error_exponent = error_exponent

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.x, y0, dy0, self.K[-1], self.direction,
                self.error_exponent, self.rtol, self.atol)
        else:
            self.h_abs = np.abs(first_step)
        self.step_size = self.direction * self.h_abs
    # ------------------------------------------------------------------
    def step(self) -> bool:
        (running,
         self.x,
         self.y,
         self.dy,
         self.h_abs,
         self.step_size) = step(self.fun,
                        self.direction,
                        self.x,
                        self.y,
                        self.dy,
                        self.x_bound,
                        self.h_abs,
                        self.max_step,
                        self.K,
                        self.K2,
                        self.n_stages,
                        self.rtol,
                        self.atol,
                        self.A,
                        self.B,
                        self.C,
                        self.E,
                        self.error_exponent)
        return running
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[np.float64, npAFloat64, npAFloat64]:
        return self.x, self.y, self.dy
# ======================================================================
@nb.njit(cache = False) # Some issue in making caching jitclasses
def init_RK2(fun: ODE2Type,
            x0: np.float64,
            y0: npAFloat64,
            dy0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params) -> _RK2:
    return _RK2(fun, x0, y0, dy0, x_bound, max_step, rtol, atol, first_step,
               *solver_params)
# ----------------------------------------------------------------------
class RK_Solver2:
    _solver_params: RK_params_type
    def __new__(cls, # type: ignore[misc]
                fun: ODE2Type,
                x0: float,
                y0: Arrayable,
                dy0: Arrayable,
                x_bound: float,
                max_step: float = np.inf,
                rtol: Arrayable = 1e-3,
                atol: Arrayable = 1e-6,
                first_step: float = 0.) -> _RK2:
        y0, dy0, rtol, atol = convert(y0, dy0, rtol, atol)
        return init_RK2(fun,
                        np.float64(x0),
                        y0,
                        dy0,
                        np.float64(x_bound),
                        np.float64(max_step),
                        rtol,
                        atol,
                        np.float64(first_step),
                        cls._solver_params)
# ----------------------------------------------------------------------
class RK23_2(RK_Solver2):
    _solver_params = RK23_params
# ----------------------------------------------------------------------
RK45_error_estimator_exponent = np.float64(-1. / (4. + 1.))
RK45_A = np.array((
            (0.,        0.,             0.,         0.,         0.),
            (1/5,       0.,             0.,         0.,         0.),
            (3/40,      9/40,           0.,         0.,         0.),
            (44/45,     -56/15,         32/9,       0.,         0.),
            (19372/6561, -25360/2187,   64448/6561, -212/729,   0.),
            (9017/3168, -355/33,        46732/5247, 49/176,     -5103/18656),
            (9017/3168, -355/33,        46732/5247, 49/176,     -5103/18656),
    ), dtype = np.float64)
RK45_B = np.array((35/384, 0, 500/1113, 125/192, -2187/6784, 11/84),
                   dtype = np.float64)
RK45_C = np.array((0, 1/5, 3/10, 4/5, 8/9, 1, 6/5), dtype = np.float64)
RK45_E = np.array((-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40),
                   dtype = np.float64)
RK45_n_stages = np.int8(len(RK45_C) - 1)
RK45_params2: RK_params_type = (RK45_error_estimator_exponent,
                 RK45_n_stages,
                 RK45_A,
                 RK45_B,
                 RK45_C,
                 RK45_E)
class RK45_2(RK_Solver2):
    _solver_params = RK45_params
# ----------------------------------------------------------------------
class Solvers2(IterableNamespace):
    RK23_2 = RK23_2
    RK45_2 = RK45_2
