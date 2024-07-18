"""Basic RK integrators implemented with numba jitclass."""
from enum import Enum

import numba as nb
import numpy as np

from ._aux import Arrayable
from ._aux import calc_eps
from ._aux import calc_tolerance
from ._aux import convert
from ._aux import IS_CACHE
from ._aux import MAX_FACTOR
from ._aux import MIN_FACTOR
from ._aux import nbA
from ._aux import nbARO
from ._aux import nbODEtype
from ._aux import norm
from ._aux import npAFloat64
from ._aux import ODEType
from ._aux import RK23_params
from ._aux import RK45_params
from ._aux import SAFETY
from ._aux import Solver
# ----------------------------------------------------------------------
try:
    from enum import member
except ImportError:
    member = lambda m: m # type: ignore
# ======================================================================
@nb.njit(fastmath = True, cache = IS_CACHE)
def calc_h0(y0: npAFloat64,
            dy0: npAFloat64,
            direction: np.float64,
            scale: npAFloat64):
    d0 = norm(y0 / scale)
    d1 = norm(dy0 / scale)

    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1

    y1 = y0 + h0 * direction * dy0
    return h0, y1, d1
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
@nb.njit(nb.float64(nbODEtype,
                    nb.float64,
                    nb.float64[:],
                    nb.float64[:],
                    nb.float64,
                    nb.float64,
                    nbARO(1),
                    nbARO(1)),
         fastmath = True, cache = IS_CACHE)
def select_initial_step(fun: ODEType,
                        x0: np.float64,
                        y0: npAFloat64,
                        dy0: npAFloat64,
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
    scale = calc_tolerance(np.abs(y0), rtol, atol)
    h0, y1, d1 = calc_h0(y0, dy0, direction, scale)
    dy1 = fun(x0 + h0 * direction, y1)
    return calc_h_abs(dy1 - dy0, h0, scale, error_exponent, d1)
# ======================================================================
@nb.njit(cache = IS_CACHE)
def step_prep(h_abs: np.float64, x: np.float64, direction: np.float64):
    x_old = x
    eps = calc_eps(x_old, direction)
    min_step = 8 * eps

    if h_abs < min_step:
        h_abs = min_step
    return h_abs, x_old, eps, min_step
# ----------------------------------------------------------------------
@nb.njit(cache = IS_CACHE)
def h_prep(h_abs: np.float64,
           max_step: np.float64,
            eps: np.float64,
            x_old: np.float64,
            x_bound: np.float64,
            direction: np.float64) -> tuple[np.float64, np.float64, np.float64]:
    if h_abs > max_step:
        h_abs = max_step - eps
    h = h_abs #* direction
    # Updating
    x = x_old + h

    if direction * (x - x_bound) >= 0: # End reached
        x = x_bound
        h = x - x_old
        h_abs = np.abs(h) # There is something weird going on here
    return x, h, h_abs
# ----------------------------------------------------------------------
@nb.njit(cache = IS_CACHE)
def calc_error_norm(K: npAFloat64,
                    E: npAFloat64,
                    h: np.float64,
                    y: np.float64,
                    y_old: npAFloat64,
                    rtol: npAFloat64,
                    atol: npAFloat64) -> np.float64:
    step_err_estimator = np.dot(K.T, E)
    step_err_estimator *= h
    y_max_abs = np.abs(y_old)
    np.maximum(y_max_abs, np.abs(y), y_max_abs)
    step_err_estimator /= calc_tolerance(y_max_abs, rtol, atol)
    return norm(step_err_estimator)
# ----------------------------------------------------------------------
# @nb.njit(nb.types.Tuple((nb.boolean,
#                          nb.float64,
#                          nb.float64[:],
#                          nb.float64,
#                          nb.float64,
#                          nbA(2)))(nbODEtype,
#                                   nb.int8,
#                                   nb.float64,
#                                   nb.float64[:],
#                                   nb.float64,
#                                   nb.float64,
#                                   nb.float64,
#                                   nbA(2),
#                                   nb.int8,
#                                   nbARO(1),
#                                   nbARO(1),
#                                   nbARO(2),
#                                   nbARO(1),
#                                   nbARO(1),
#                                   nbARO(1),
#                                   nb.float64),
#         cache = IS_CACHE)
@nb.njit(cache = IS_CACHE)
def step(fun: ODEType,
          direction: np.float64,
          x: np.float64,
          y: npAFloat64,
          x_bound: np.float64,
          h_abs: np.float64,
          max_step: np.float64,
          K: npAFloat64,
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
        return False, x, y, h_abs, direction *h_abs, K

    h_abs, x_old, eps, min_step = step_prep(h_abs, x, direction)
    y_old = y

    while True: # := not working
        x, h, h_abs = h_prep(h_abs, max_step, eps, x_old, x_bound, direction)

        # RK core loop
        K[0] = K[-1]
        for s in range(1, n_stages):
            K[s] = fun(x_old + C[s] * h,
                       y_old + np.dot(K[:s].T, A[s,:s]) * h)

        y = y_old + h * np.dot(K[:-1].T, B)

        K[-1] = fun(x, y)

        error_norm = calc_error_norm(K, E, h, y, y_old, rtol, atol)

        if error_norm < 1.:
            h_abs *= (MAX_FACTOR if error_norm == 0. else
                      min(MAX_FACTOR, SAFETY * error_norm ** error_exponent))
            return True, x, y, h_abs, h, K # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** error_exponent)
            if h_abs < min_step:
                return False, x, y, h_abs, h, K # Too small step size
# ----------------------------------------------------------------------
base_spec = (('A', nbARO(2)),
             ('B', nbARO(1)),
             ('C', nbARO(1)),
             ('E', nbARO(1)),
             ('K', nbA(2)),
             ('n_stages', nb.int8),
             ('x', nb.float64),
             ('y', nb.float64[:]),
             ('x_bound', nb.float64),
             ('direction', nb.float64),
             ('max_step', nb.float64),
             ('error_exponent', nb.float64),
             ('h_abs', nb.float64),
             ('step_size', nb.float64),
             ('atol', nbARO(1)),
             ('rtol', nbARO(1)))
# ----------------------------------------------------------------------
@nb.experimental.jitclass(base_spec + (('fun', nbODEtype),))
class RK(Solver):
    """Base class for explicit Runge-Kutta methods."""

    def __init__(self,
                 fun: ODEType,
                 x0: np.float64,
                 y0: npAFloat64,
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
        self.x_bound = x_bound
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step

        self.K = np.zeros((self.n_stages + 1, len(y0)), dtype = self.y.dtype)
        self.K[-1] = self.fun(self.x, self.y)
        self.direction = 1. if x_bound == x0 else np.sign(x_bound - x0)
        self.error_exponent = error_exponent

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.x, y0, self.K[-1], self.direction,
                self.error_exponent, self.rtol, self.atol)
        else:
            self.h_abs = np.abs(first_step)
        self.step_size = self.direction * self.h_abs
    # ------------------------------------------------------------------
    def step(self) -> bool:
        (running,
         self.x,
         self.y,
         self.h_abs,
         self.step_size,
         self.K) = step(self.fun,
                        self.direction,
                        self.x,
                        self.y,
                        self.x_bound,
                        self.h_abs,
                        self.max_step,
                        self.K,
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
    def state(self) -> tuple[np.float64, npAFloat64]:
        return self.x, self.y
# ======================================================================
@nb.njit(cache = False) # Some issue in making caching jitclasses
def init_RK(fun: ODEType,
            x0: np.float64,
            y0: npAFloat64,
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
            E: npAFloat64) -> RK:
    return RK(fun,
              x0,
              y0,
              x_bound,
              max_step,
              rtol,
              atol,
              first_step,
              error_exponent,
              n_stages,
              A,
              B,
              C,
              E)
# ----------------------------------------------------------------------
def RK23(fun: ODEType,
         x0: float,
         y0: Arrayable,
         x_bound: float,
         max_step: float = np.inf,
         rtol: Arrayable = 1e-3,
         atol: Arrayable = 1e-6,
         first_step: float = 0) -> RK:

    y0, rtol, atol = convert(y0, rtol, atol)
    return init_RK(fun,
                       np.float64(x0),
                       y0,
                       np.float64(x_bound),
                       np.float64(max_step),
                       rtol,
                       atol,
                       np.float64(first_step),
                       *RK23_params)
# ----------------------------------------------------------------------
def RK45(fun: ODEType,
         x0: float,
         y0: Arrayable,
         x_bound: float,
         max_step: float = np.inf,
         rtol: Arrayable = 1e-3,
         atol: Arrayable = 1e-6,
         first_step: float = 0.) -> RK:

    y0, rtol, atol = convert(y0, rtol, atol)
    return init_RK(fun,
                   np.float64(x0),
                   y0,
                   np.float64(x_bound),
                   np.float64(max_step),
                   rtol,
                   atol,
                   np.float64(first_step),
                   *RK23_params)
# ======================================================================
ALL = (RK23, RK45)
class Solvers(Enum):
    RK23 = member(RK23)
    RK45 = member(RK45)
    ALL = member(ALL)
