"""Basic RK integrators implemented with numba jitclass."""
import numba as nb
import numpy as np

from .._aux import Arrayable
from .._aux import calc_error
from .._aux import calc_tolerance
from .._aux import convert
from .._aux import IS_CACHE
from .._aux import IterableNamespace
from .._aux import jitclass_from_dict
from .._aux import MAX_FACTOR
from .._aux import MIN_FACTOR
from .._aux import nbA
from .._aux import nbARO
from .._aux import nbDec
from .._aux import nbDecC
from .._aux import npAFloat64
from .._aux import RK23_params
from .._aux import RK45_params
from .._aux import RK_Params
from .._aux import SAFETY
from ._first_aux import calc_h0
from ._first_aux import calc_h_abs
from ._first_aux import FirstSolverBase
from ._first_aux import nbODE_type
from ._first_aux import ODE1Type
# ======================================================================
@nbDec(nb.float64(nbODE_type,
                    nb.float64,
                    nbA(1),
                    nbA(1),
                    nb.float64,
                    nb.float64,
                    nbARO(1),
                    nbARO(1)),
         fastmath = True, cache = IS_CACHE)
def select_initial_step(fun: ODE1Type,
                        x0: np.float64,
                        y0: npAFloat64,
                        dy0: npAFloat64,
                        direction: np.float64,
                        error_exponent: np.float64,
                        rtol: npAFloat64,
                        atol: npAFloat64) -> np.float64:
    """Empirically select a good initial _RK_adaptive_step.

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
        Absolute value of the suggested initial _RK_adaptive_step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    scale = calc_tolerance(np.abs(y0), rtol, atol)
    h, h_abs, y1, d1 = calc_h0(y0, dy0, direction, scale)
    return calc_h_abs(fun(x0 + h, y1) - dy0, h_abs, scale, error_exponent, d1)
# ======================================================================
@nbDecC
def _step(fun: ODE1Type,
             x0: np.float64,
             y0: npAFloat64,
             h: np.float64,
             K: npAFloat64,
             n_stages: np.int64,
             A: npAFloat64,
             C: npAFloat64) -> tuple[np.float64, np.float64, npAFloat64]:
    """RK core step."""
    Dx = C[0] * h
    K[1] = fun(x0 + Dx, y0 + Dx * K[0])

    for s in range(2, n_stages):
        Dx = C[s - 1] * h
        K[s] = fun(x0 + Dx,
                    y0 + np.dot(Dx * A[s - 2,:s], K[:s]))

    # Last substep
    x = x0 + h
    y = y0 + np.dot(A[-2,:-1] * h, K[:-1])
    K[-1] = fun(x, y)

    return x, y
# ======================================================================
@nbDecC
def _RK_adaptive_step(fun: ODE1Type,
          x0: np.float64,
          y0: npAFloat64,
          h: np.float64,
          h_abs_min: np.float64,
          K: npAFloat64,
          rtol: npAFloat64,
          atol: npAFloat64,
          _len: np.float64,
          n_stages: np.int64,
          A: npAFloat64,
          C: npAFloat64,
          error_exponent: np.float64
          ) -> tuple[np.float64, np.float64, npAFloat64, np.float64, np.int64]:
    nfev = np.int64(n_stages)
    y0_abs = np.abs(y0)
    x, y = _step(fun, x0, y0, h, K, n_stages, A, C)
    error = calc_error(np.dot(A[-1], K), y, y0_abs, rtol, atol) * _len * h * h

    while error > 1. and abs(h) > h_abs_min: # := not working
        h *= max(MIN_FACTOR, SAFETY * error ** error_exponent)
        x, y = _step(fun, x0, y0, h, K, n_stages, A, C)
        error = calc_error(np.dot(A[-1], K), y, y0_abs, rtol, atol
                           ) * _len * h * h
        nfev += n_stages
    return error, x, y, h, nfev
# ----------------------------------------------------------------------
@jitclass_from_dict({'fun': nbODE_type})
class _RK(FirstSolverBase):
    """Base class for explicit Runge-Kutta methods."""

    def __init__(self,
                 fun: ODE1Type,
                 x0: np.float64,
                 y0: npAFloat64,
                 x_bound: np.float64,
                 max_step: np.float64,
                 rtol: npAFloat64,
                 atol: npAFloat64,
                 first_step: np.float64,
                 n_stages: np.int64,
                 A: npAFloat64,
                 C: npAFloat64,
                 error_exponent: np.float64):
        self.fun = fun
        self.x = x0
        self.y = y0
        self.x_bound = x_bound
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.n_stages = n_stages
        self.A = A
        self.C = C
        self.error_exponent = error_exponent

        self.reboot(first_step)
    # ------------------------------------------------------------------
    def _init_K(self, y_len: int):
        self.K = np.zeros((self.n_stages + np.int64(1), y_len), np.float64)
        self.K[0] = self.fun(self.x, self.y)
    # ------------------------------------------------------------------
    def _select_initial_step(self, direction: np.float64) -> np.float64:
        return select_initial_step(self.fun, self.x, self.y, self.K[-1],
                    direction, self.error_exponent, self.rtol, self.atol)
    # ------------------------------------------------------------------
    def _step(self) -> tuple[np.float64, np.int64]:
        (error,
        self.x,
        self.y,
        self.step_size,
        nfev) = _RK_adaptive_step(self.fun,
                    self.x,
                    self.y,
                    self._h_next,
                    self._h_abs_min,
                    self.K,
                    self.rtol,
                    self.atol,
                    self._len,
                    self.n_stages,
                    self.A,
                    self.C,
                    self.error_exponent)
        self.K[0] = self.K[-1]
        return  error, nfev
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[np.float64, npAFloat64, npAFloat64]:
        return self.x, self.y, self.K[0]
# ======================================================================
@nbDec(cache = False) # Some issue in making caching jitclasses
def init_RK(fun: ODE1Type,
            x0: np.float64,
            y0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params: tuple) -> _RK:
    return _RK(fun, x0, y0, x_bound, max_step, rtol, atol, first_step,
               *solver_params)
# ----------------------------------------------------------------------
class RK_Solver:
    _solver_params: RK_Params
    def __new__(cls, # type: ignore[misc]
                fun: ODE1Type,
                x0: float,
                y0: Arrayable,
                x_bound: float,
                *,
                max_step: float = np.inf,
                rtol: Arrayable = 1e-6,
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
