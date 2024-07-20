"""Basic RK integrators implemented with numba jitclass."""
import numba as nb
import numpy as np

from .._aux import Arrayable
from .._aux import calc_eps
from .._aux import calc_error_norm2
from .._aux import calc_tolerance
from .._aux import convert
from .._aux import IS_CACHE
from .._aux import IterableNamespace
from .._aux import jitclass_from_dict
from .._aux import MAX_FACTOR
from .._aux import MIN_FACTOR
from .._aux import nbA
from .._aux import nbARO
from .._aux import nbDecC
from .._aux import nbODE_type
from .._aux import npAFloat64
from .._aux import ODEType
from .._aux import RK23_params
from .._aux import RK45_params
from .._aux import RK_Params
from .._aux import SAFETY
from .._aux import SolverBase
from ._first_aux import calc_h0
from ._first_aux import calc_h_abs
# ======================================================================
@nb.njit(nb.float64(nbODE_type,
                    nb.float64,
                    nbA(1),
                    nbA(1),
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
@nbDecC
def step_prep(h_abs: np.float64, x: np.float64, direction: np.float64):
    x_old = x
    eps = calc_eps(x_old, direction)
    min_step = 8 * eps

    if h_abs < min_step:
        h_abs = min_step
    return h_abs, x_old, eps, min_step
# ----------------------------------------------------------------------
@nbDecC
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
@nbDecC
def step(fun: ODEType,
          direction: np.float64,
          x: np.float64,
          y: npAFloat64,
          x_bound: np.float64,
          h_abs: np.float64,
          h: np.float64,
          max_step: np.float64,
          K: npAFloat64,
          n_stages: np.uint64,
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
                                          np.uint64]:
    if direction * (x - x_bound) >= 0: # x_bound has been reached
        return False, x, y, h_abs, h, np.uint64(0)

    h_abs, x_old, eps, min_step = step_prep(h_abs, x, direction)
    y_old = y
    K[0] = K[-1]
    nfev = np.uint64(0)
    while True: # := not working
        x, h, h_abs = h_prep(h_abs, max_step, eps, x_old, x_bound, direction)

        # _RK core loop
        # len(K) == n_stages + 1
        # First step simply

        K[1] = fun(x_old + C[1] * h,
                   y_old + A[1,0] * h * K[0])
        for s in range(2, n_stages):
            K[s] = fun(x_old + C[s] * h,
                       y_old + np.dot(A[s,:s] * h, K[:s]))

        y = y_old + h * np.dot(B, K[:-1])

        K[-1] = fun(x, y)

        nfev += n_stages

        error_norm2 = calc_error_norm2(K, E, h, y, y_old, rtol, atol)

        if error_norm2 < 1.:
            h_abs *= (MAX_FACTOR if error_norm2 == 0. else
                      min(MAX_FACTOR, SAFETY * error_norm2 ** error_exponent))
            return True, x, y, h_abs, h, nfev # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm2 ** error_exponent)
            if h_abs < min_step:
                return False, x, y, h_abs, h, nfev# Too small step size
# ----------------------------------------------------------------------
@jitclass_from_dict()
class _RK(SolverBase):
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
                 n_stages: np.uint64,
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

        self.K = np.zeros((self.n_stages + np.uint64(1), len(y0)), dtype = y0.dtype)
        self.K[-1] = self.fun(self.x, self.y)
        self._nfev = np.uint64(1)
        self.direction = 1. if x_bound == x0 else np.sign(x_bound - x0)
        self.error_exponent = error_exponent

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.x, y0, self.K[-1], self.direction,
                self.error_exponent, self.rtol, self.atol)
            self._nfev += np.uint64(1)
        else:
            self.h_abs = np.abs(first_step)
        self.step_size = 0.
    # ------------------------------------------------------------------
    def step(self) -> bool:
        (running,
         self.x,
         self.y,
         self.h_abs,
         self.step_size,
         nfev) = step(self.fun,
                        self.direction,
                        self.x,
                        self.y,
                        self.x_bound,
                        self.h_abs,
                        self.step_size,
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
        self._nfev += nfev
        return running
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[np.float64, npAFloat64]:
        return self.x, self.y
# ======================================================================
# @nb.njit(cache = False) # Some issue in making caching jitclasses
def init_RK(fun: ODEType,
            x0: np.float64,
            y0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params) -> _RK:
    return _RK(fun, x0, y0, x_bound, max_step, rtol, atol, first_step,
               *solver_params)
# ----------------------------------------------------------------------
class RK_Solver:
    _solver_params: RK_Params
    def __new__(cls, # type: ignore[misc]
                fun: ODEType,
                x0: float,
                y0: Arrayable,
                x_bound: float,
                *,
                max_step: float = np.inf,
                rtol: Arrayable = 1e-3,
                atol: Arrayable = 1e-6,
                first_step: float = 0.) -> _RK:
        y0, rtol, atol = convert(y0, rtol, atol)
        return init_RK(fun,
                        np.float64(x0),
                        y0,
                        np.float64(x_bound),
                        np.float64(max_step),
                        rtol,
                        atol,
                        np.float64(first_step),
                        tuple(cls._solver_params))
# ----------------------------------------------------------------------
class RK23(RK_Solver):
    _solver_params = RK23_params
# ----------------------------------------------------------------------
class RK45(RK_Solver):
    _solver_params = RK45_params
# ======================================================================
class Solvers(IterableNamespace):
    RK23 = RK23
    RK45 = RK45
