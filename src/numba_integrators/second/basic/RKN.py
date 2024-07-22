"""Basic RK integrators implemented with numba jitclass."""
from collections.abc import Iterable
from typing import NamedTuple

import numba as nb
import numpy as np

from ..._aux import Arrayable
from ..._aux import calc_eps
from ..._aux import calc_error_norm2
from ..._aux import calc_tolerance
from ..._aux import convert
from ..._aux import IS_CACHE
from ..._aux import IterableNamespace
from ..._aux import jitclass_from_dict
from ..._aux import MAX_FACTOR
from ..._aux import MIN_FACTOR
from ..._aux import nbA
from ..._aux import nbARO
from ..._aux import nbDec
from ..._aux import nbDecC
from ..._aux import nbDecFC
from ..._aux import nbODE2_type
from ..._aux import norm
from ..._aux import npAFloat64
from ..._aux import SAFETY
from ..._aux import SolverBase
from ..._aux import step_prep
from .._second_aux import RKN56_params
from ._second_basic_aux import ODE2Type
from ._second_basic_aux import select_initial_step
from ._second_basic_aux import Solver2
# ----------------------------------------------------------------------
@nbDecC
def RKNstep(fun: ODE2Type,
          direction: np.float64,
          x_old: np.float64,
          y_old: npAFloat64,
          y_prev: npAFloat64,
          h_abs: np.float64,
          h_prev: np.float64,
          min_step: np.float64,
          K: npAFloat64,
          n_stages: np.uint64,
          rtol: npAFloat64,
          atol: npAFloat64,
          alpha: npAFloat64,
          beta: npAFloat64,
          gamma: npAFloat64,
          error_exponent: np.float64) -> tuple[bool,
                                          np.float64,
                                          npAFloat64,
                                          np.float64,
                                          np.float64,
                                          np.uint64]:
    nfev = np.uint64(0)
    # Prepping ofor next step
    K[0, 0] = K[0, -1]
    K[1, 0] = K[1, -1]

    y_old_abs = np.abs(y_old)
    dy_old_abs = np.abs(K[0, 0])
    _len = 1. / len(y_old)

    dy0 = K[0, 0]
    ddy0 = K[1, 0]
    while True: # := not working
        h = direction * h_abs

        # First substep
        Dx = alpha[0] * h
        x = x_old + Dx
        dy = ddy0 * Dx
        y = dy0 + Dx * ((0.5 * Dx) * ddy0 + dy0
        ( ), ddy0 * Dx)


        nfev += n_stages
        # Error calculation
        h2 = h*h
        error_norm2 = (calc_error_norm2(
            np.dot(A[-1], K[0]), np.abs(y), y_old_abs, rtol, atol)
                       + calc_error_norm2(
            np.dot(A[-1], K[1]), np.abs(K[0, -1]), dy_old_abs, rtol, atol)
            ) * 0.5 * _len * h2

        if error_norm2 < 1.:
            h_abs *= (MAX_FACTOR if error_norm2 == 0. else
                      min(MAX_FACTOR, SAFETY * error_norm2 ** error_exponent))
            if h_abs < min_step: # Due to the SAFETY, the step can shrink
                h_abs = min_step
            return True, y, x_old + h, h_abs, h, nfev # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm2 ** error_exponent)
            if h_abs < min_step:
                return False, y, x_old + h, h_abs, h, nfev # Too small step size
# ----------------------------------------------------------------------
@jitclass_from_dict({'fun': nbODE2_type,
                     'K': nbA(3),
                     'alpha': nbA(1),
                     'beta': nbA(2),
                     'gamma': nbA(2)}, ('A', 'C'))
class _RKN(SolverBase):
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
                 n_stages: np.uint64,
                 alpha: npAFloat64,
                 beta: npAFloat64,
                 gamma:  npAFloat64):
        self.n_stages = n_stages
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fun = fun
        self.x = x0
        self.y = y0
        self.x_bound = x_bound
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.error_exponent = error_exponent

        _1 = np.uint64(1)

        self.K = np.zeros((2, self.n_stages + _1, len(y0)), dtype = np.float64)
        self.K[0, -1] = dy0
        self.K[1, -1] = self.fun(self.x, self.y, self.dy)
        self._nfev = _1
        self.direction = 1. if x_bound == x0 else np.sign(x_bound - x0)

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.x, y0, dy0, self.K[1, -1], self.direction,
                self.error_exponent, self.rtol, self.atol)
            self._nfev += _1
        else:
            self.h_abs = np.abs(first_step)

        min_step = 8. * calc_eps(self.x, self.direction)
        if self.h_abs < min_step:
            self.h_abs = min_step

        self.step_size = 0.
    # ------------------------------------------------------------------
    def step(self) -> bool:
        h_end = self.x_bound - self.x
        if self.direction * h_end > 0.: # x_bound has not been reached
            min_step, h_abs = step_prep(self.x,
                                        self.h_abs,
                                        self.direction,
                                        h_end,
                                        self.max_step)
            (valid,
            self.y,
            self.x,
            self.h_abs,
            self.step_size,
            nfev) = RKNstep(self.fun,
                            self.direction,
                            self.x,
                            self.y,
                            h_abs,
                            min_step,
                            self.K,
                            self.n_stages,
                            self.rtol,
                            self.atol,
                            self.alpha,
                            self.beta,
                            self.gamma,
                            self.error_exponent)
            self._nfev += nfev
            return valid
        return False
    # ------------------------------------------------------------------
    @property
    def dy(self) -> npAFloat64:
        return self.K[0, -1]
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[np.float64, npAFloat64, npAFloat64]:
        return self.x, self.y, self.K[0, -1]
# ======================================================================
@nbDec(cache = False) # Some issue in making caching jitclasses
def init_RKN(fun: ODE2Type,
            x0: np.float64,
            y0: npAFloat64,
            dy0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params) -> _RKN:
    return _RKN(fun, x0, y0, dy0, x_bound, max_step, rtol, atol, first_step,
               *solver_params)
# ----------------------------------------------------------------------
class RKN_Solver(Solver2):
    _init = init_RKN
# ----------------------------------------------------------------------
class RKN56(RKN_Solver):
    _solver_params = RKN56_params
# ----------------------------------------------------------------------
class Solvers(IterableNamespace):
    RKN56 = RKN56
