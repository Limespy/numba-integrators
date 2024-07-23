"""Basic RK integrators implemented with numba jitclass."""
import numba as nb
import numpy as np

from ..._aux import calc_error
from ..._aux import IterableNamespace
from ..._aux import jitclass_from_dict
from ..._aux import MIN_FACTOR
from ..._aux import nbA
from ..._aux import nbARO
from ..._aux import nbDec
from ..._aux import nbDecC
from ..._aux import npAFloat64
from ..._aux import SAFETY
from .._second_aux import RKN56_params
from ._second_basic_aux import nbODE2_type
from ._second_basic_aux import ODE2Type
from ._second_basic_aux import SecondBasicSolverBase
from ._second_basic_aux import select_initial_step
from ._second_basic_aux import Solver2
# ======================================================================
def _step(fun, x0, y0, dy0, h, K, n_stages, alpha, beta, gamma):

    ddy0 = K[0]

    # first step
    Dx = alpha[0] * h
    x = x0 + Dx
    dy = dy0 + ddy0 * Dx
    y = y0 + Dx * ((0.5 * Dx) * ddy0 + dy0)

    K[1] = fun(x, y, dy)

    # Second step
    for s in range(2, n_stages-1):
        Dx = alpha[s - 1] * h

        x = x0 + Dx
        dy = dy0 + np.dot(Dx * beta[s - 2,:s], K[:s])
        y = y0 + Dx * (np.dot(Dx * gamma[s-2, :s], K[:s]) + dy0)

        K[s] = fun(x, y, dy)

    # last steps at h
    x = x0 + h

    # second to last step
    dy = dy0 + np.dot(h * beta[-2,:-1], K[:-2])
    y = y0 + h * (np.dot(h * gamma[-2, :-1], K[:-2]) + dy0)

    K[-2] = fun(x, y, dy)


    # last step
    dy = dy0 + np.dot(h * beta[-1], K[:-1])
    y = y0 + h * (np.dot(h * gamma[-1], K[:-1]) + dy0)

    K[-1] = fun(x, y, dy)


    return x, y, dy
# ----------------------------------------------------------------------
@nbDecC
def _RKNFstep(fun: ODE2Type,
          x0: np.float64,
          y0: npAFloat64,
          dy0: npAFloat64,
          h: np.float64,
          h_abs_min: np.float64,
          K: npAFloat64,
          rtol: npAFloat64,
          atol: npAFloat64,
          _len: np.float64,
          n_stages: np.int64,
          alpha: npAFloat64,
          beta: npAFloat64,
          gamma: npAFloat64,
          error_exponent: np.float64) -> tuple[bool,
                                          np.float64,
                                          npAFloat64,
                                          np.float64,
                                          np.float64,
                                          np.int64]:

    # Prepping for next step
    nfev = np.int64(n_stages)
    y0_abs = np.abs(y0)

    x, y, dy = _step(fun, x0, y0, dy0, h, K, n_stages, alpha, beta, gamma)
    h2 = h * h
    error = calc_error(gamma[-1,-1] * (K[-2] - K[-1]) , y, y0_abs, rtol, atol
                       ) * _len * h2 * h2
    n_retrys = 0
    while error > 1. and abs(h) > h_abs_min: # := not working
        h *= max(MIN_FACTOR, SAFETY * error ** error_exponent)
        x, y, dy = _step(fun, x0, y0, dy0, h, K, n_stages, alpha, beta, gamma)

        h2 = h * h
        error = calc_error(gamma[-1,-1] * (K[-2] - K[-1]) ,
                           y, y0_abs, rtol, atol) * _len * h2 * h2
        nfev += n_stages
        n_retrys += 1
    return error, x, y, dy, h, nfev
# ----------------------------------------------------------------------
@jitclass_from_dict({'fun': nbODE2_type,
                     'dy': nbA(1),
                     'alpha': nbA(1),
                     'beta': nbA(2),
                     'gamma': nbA(2)},
                     ('A', 'C'))
class _RKNF(SecondBasicSolverBase):
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
                 n_stages: np.int64,
                 alpha: npAFloat64,
                 beta: npAFloat64,
                 gamma:  npAFloat64,
                 error_exponent: np.float64,):
        self.fun = fun
        self.x = x0
        self.y = y0
        self.dy = dy0
        self.x_bound = x_bound
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.n_stages = n_stages
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.error_exponent = error_exponent

        self.reboot(first_step)
    # ------------------------------------------------------------------
    def _init_K(self, y_len: int):
        self.K = np.zeros((self.n_stages + np.int64(1), y_len), np.float64)
        self.K[0] = self.fun(self.x, self.y, self.dy)
    # ------------------------------------------------------------------
    def _select_initial_step(self, direction: np.float64) -> np.float64:
        return select_initial_step(self.fun, self.x, self.y, self.dy,
                                   self.K[0], direction,
                                   self.error_exponent, self.rtol, self.atol)
    # ------------------------------------------------------------------
    def _step(self) -> tuple[np.float64, np.int64]:
        (error,
        self.x,
        self.y,
        self.dy,
        self.step_size,
        nfev) = _RKNFstep(self.fun,
                        self.x,
                        self.y,
                        self.dy,
                        self._h_next,
                        self._h_abs_min,
                        self.K,
                        self.rtol,
                        self.atol,
                        self._len,
                        self.n_stages,
                        self.alpha,
                        self.beta,
                        self.gamma,
                        self.error_exponent)
        self.K[0] = self.K[-1]
        return error, nfev
    # ------------------------------------------------------------------
    @property
    def ddy(self) -> npAFloat64:
        return self.K[0]
# ======================================================================
@nbDec(cache = False) # Some issue in making caching jitclasses
def init_RKNF(fun: ODE2Type,
            x0: np.float64,
            y0: npAFloat64,
            dy0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params) -> _RKNF:
    return _RKNF(fun, x0, y0, dy0, x_bound, max_step, rtol, atol, first_step,
               *solver_params)
# ----------------------------------------------------------------------
class RKNF_Solver(Solver2):
    _init = init_RKNF
# ----------------------------------------------------------------------
class RKNF56(RKNF_Solver):
    _solver_params = RKN56_params
# ----------------------------------------------------------------------
class Solvers(IterableNamespace):
    RKNF56 = RKNF56
