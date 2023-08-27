'Basic integrators'
import enum
from typing import Iterable
from typing import Union

import numba as nb
import numpy as np

from ._aux import convert
from ._aux import IS_CACHE
from ._aux import MAX_FACTOR
from ._aux import MIN_FACTOR
from ._aux import nbA
from ._aux import nbARO
from ._aux import nbODEtype
from ._aux import norm
from ._aux import npAFloat64
from ._aux import ODEFUN
from ._aux import RK23_params
from ._aux import RK45_params
from ._aux import SAFETY
# ----------------------------------------------------------------------
@nb.njit(nb.float64(nbODEtype,
                    nb.float64,
                    nb.float64[:],
                    nb.float64[:],
                    nb.int8,
                    nb.float64,
                    nbARO(1),
                    nbARO(1)),
         fastmath = True, cache = IS_CACHE)
def select_initial_step(fun: ODEFUN,
                        t0: np.float64,
                        y0: npAFloat64,
                        f0: npAFloat64,
                        direction: np.float64,
                        error_estimator: np.float64,
                        rtol: npAFloat64,
                        atol: npAFloat64) -> np.float64:
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : np.float64
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : np.float64
        Integration direction.
    order : np.float64
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

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)

    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    h1 = (max(1e-6, h0 * 1e-3) if d1 <= 1e-15 and d2 <= 1e-15
          else (max(d1, d2) * 100) ** error_estimator)

    return min(100 * h0, h1)
# ----------------------------------------------------------------------
@nb.njit(nb.types.Tuple((nb.boolean,
                         nb.float64,
                         nb.float64[:],
                         nb.float64,
                         nb.float64,
                         nbA(2)))(nbODEtype,
                                  nb.int8,
                                  nb.float64,
                                  nb.float64[:],
                                  nb.float64,
                                  nb.float64,
                                  nb.float64,
                                  nbA(2),
                                  nb.int8,
                                  nbARO(1),
                                  nbARO(1),
                                  nbARO(2),
                                  nbARO(1),
                                  nbARO(1),
                                  nbARO(1),
                                  nb.float64),
        cache = IS_CACHE)
def _step(fun: ODEFUN,
          direction: np.float64,
          t: np.float64,
          y: npAFloat64,
          t_bound: np.float64,
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
    if direction * (t - t_bound) >= 0: # t_bound has been reached
        return False, t, y, h_abs, direction *h_abs, K
    t_old = t
    y_old = y
    min_step = 10. * np.abs(np.nextafter(t, direction * np.inf) - t)

    if h_abs < min_step:
        h_abs = min_step

    while True: # := not working
        if h_abs > max_step:
            h_abs = max_step
        h = h_abs * direction
        # Updating
        t = t_old + h

        K[0] = K[-1]

        if direction * (t - t_bound) >= 0:
            t = t_bound
            h = t - t_old
            h_abs = np.abs(h) # There is something weird going on here
        # RK core loop
        for s in range(1, n_stages):
            K[s] = fun(t_old + C[s] * h,
                       y_old + np.dot(K[:s].T, A[s,:s]) * h)

        y = y_old + h * np.dot(K[:-1].T, B)

        K[-1] = fun(t, y)

        error_norm = norm(np.dot(K.T, E)
                          * h
                          / (atol + np.maximum(np.abs(y_old),
                                              np.abs(y)) * rtol))

        if error_norm < 1:
            h_abs *= (MAX_FACTOR if error_norm == 0 else
                            min(MAX_FACTOR,
                                SAFETY * error_norm ** error_exponent))
            return True, t, y, h_abs, h, K # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR,
                                SAFETY * error_norm ** error_exponent)
            if h_abs < min_step:
                return False, t, y, h_abs, h, K # Too small step size
# ----------------------------------------------------------------------
base_spec = (('A', nbARO(2)),
             ('B', nbARO(1)),
             ('C', nbARO(1)),
             ('E', nbARO(1)),
             ('K', nbA(2)),
             ('n_stages', nb.int8),
             ('t', nb.float64),
             ('y', nb.float64[:]),
             ('t_bound', nb.float64),
             ('direction', nb.float64),
             ('max_step', nb.float64),
             ('error_exponent', nb.float64),
             ('h_abs', nb.float64),
             ('step_size', nb.float64),
             ('atol', nbARO(1)),
             ('rtol', nbARO(1)))
# ----------------------------------------------------------------------
@nb.experimental.jitclass(base_spec + (('fun', nbODEtype),))
class RK:
    """Base class for explicit Runge-Kutta methods."""

    def __init__(self,
                 fun: ODEFUN,
                 t0: float,
                 y0: npAFloat64,
                 t_bound: float,
                 max_step: float,
                 rtol: npAFloat64,
                 atol: npAFloat64,
                 first_step: float,
                 error_estimator_order: np.int8,
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
        self.t = t0
        self.y = y0
        self.t_bound = t_bound
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step

        self.K = np.zeros((self.n_stages + 1, len(y0)), dtype = self.y.dtype)
        self.K[-1] = self.fun(self.t, self.y) # type: ignore
        self.direction = np.float64(np.sign(t_bound - t0) if t_bound != t0 else 1)
        self.error_exponent = -1 / (error_estimator_order + 1)

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.t, y0, self.K[-1], self.direction,
                self.error_exponent, self.rtol, self.atol)
        else:
            self.h_abs = np.abs(first_step)
        self.step_size = self.direction * self.h_abs
    # ------------------------------------------------------------------
    def step(self) -> bool:
        (running,
         self.t,
         self.y,
         self.h_abs,
         self.step_size,
         self.K) = _step(self.fun,
                        self.direction,
                        self.t,
                        self.y,
                        self.t_bound,
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
# ======================================================================
@nb.njit(cache = False) # Some issue' in making caching jitclasses
def RK23_direct(fun: ODEFUN,
                t0: float,
                y0: npAFloat64,
                t_bound: float,
                max_step: float,
                rtol: npAFloat64,
                atol: npAFloat64,
                first_step: float) -> RK:
    return RK(fun, t0, y0, t_bound, max_step, rtol, atol, first_step,
              *RK23_params)
# ----------------------------------------------------------------------
def RK23(fun: ODEFUN,
         t0: float,
         y0: Union[int, float, npAFloat64, Iterable],
         t_bound: float,
         max_step: float = np.inf,
         rtol: Union[int, float, npAFloat64, Iterable] = 1e-3,
         atol: Union[int, float, npAFloat64, Iterable] = 1e-6,
         first_step: float = 0) -> RK:

    y0, rtol, atol = convert(y0, rtol, atol)
    return RK23_direct(fun, t0, y0, t_bound, max_step, rtol, atol, first_step)
# ----------------------------------------------------------------------
@nb.njit(cache = False) # Some issue' in making caching jitclasses
def RK45_direct(fun: ODEFUN,
                t0: float,
                y0: npAFloat64,
                t_bound: float,
                max_step: float,
                rtol: npAFloat64,
                atol: npAFloat64,
                first_step: float) -> RK:
    return RK(fun, t0, y0, t_bound, max_step, rtol, atol, first_step,
              *RK45_params)
# ----------------------------------------------------------------------
def RK45(fun: ODEFUN,
         t0: float,
         y0: Union[int, float, npAFloat64, Iterable],
         t_bound: float,
         max_step: float = np.inf,
         rtol: Union[int, float, npAFloat64, Iterable] = 1e-3,
         atol: Union[int, float, npAFloat64, Iterable] = 1e-6,
         first_step: float = 0.) -> RK:

    y0, rtol, atol = convert(y0, rtol, atol)
    return RK45_direct(fun, t0, y0, t_bound, max_step, rtol, atol, first_step)
# ----------------------------------------------------------------------
ALL = (RK23, RK45)
class Solver(enum.Enum):
    RK23 = RK23
    RK45 = RK45
    ALL = ALL
