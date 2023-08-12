import enum
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Union

import numba as nb
import numpy as np

from ._aux import nbA
from ._aux import nbAdvanced_initial_step_signature
from ._aux import nbAdvanced_ODE_signature
from ._aux import nbAdvanced_step_signature
from ._aux import nbARO
from ._aux import nbODEtype
from ._aux import npAFloat64
from ._aux import ODEFUN
from ._aux import ODEFUNA
# ======================================================================
# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

IS_CACHE = True
# ----------------------------------------------------------------------
@nb.njit(nb.float64(nb.float64[:]),
         fastmath = True, cache = IS_CACHE)
def norm(x: npAFloat64) -> np.float64:
    """Compute RMS norm."""
    return np.sqrt(np.sum(x * x) / x.size)
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
def _into_1d_typearray(item: Union[int, float, npAFloat64, Iterable],
                       length: int = 1,
                       dtype: type = np.float64):
    if isinstance(item, np.ndarray):
        if item.ndim == 0:
            return np.full(length, item, dtype = dtype)
        elif item.ndim == 1:
            return np.asarray(item, dtype = dtype)
        else:
            raise ValueError(f'Dimensionality of y0 is over 1. y0 = {item}')
    elif isinstance(item, Iterable): # Re-checking the item as np array
        return _into_1d_typearray(np.array(item, dtype = dtype),
                                  length,
                                  dtype)
    else:
        return np.full(length, item, dtype = dtype)
# ----------------------------------------------------------------------
def convert(y0: Union[int, float, npAFloat64, Iterable],
            rtol: Union[int, float, npAFloat64, Iterable],
            atol: Union[int, float, npAFloat64, Iterable]
            ) -> tuple[npAFloat64, npAFloat64, npAFloat64]:
    '''Converts y0 and tolerances into correct type of arrays'''
    y0 = _into_1d_typearray(y0)
    return (y0,
            _into_1d_typearray(rtol, len(y0)),
            _into_1d_typearray(atol, len(y0)))
# ======================================================================
_RK23_error_estimator_order = np.int8(2)
_RK23_n_stages = np.int8(3)
_RK23_A = np.array((
    (0, 0, 0),
    (1/2, 0, 0),
    (0, 3/4, 0)
), dtype = np.float64)
_RK23_B = np.array((2/9, 1/3, 4/9), dtype = np.float64)
_RK23_C = np.array((0, 1/2, 3/4), dtype = np.float64)
_RK23_E = np.array((5/72, -1/12, -1/9, 1/8), dtype = np.float64)
# _RK23_P = np.array(((1, -4 / 3, 5 / 9), # unused
#                 (0, 1, -2/3),
#                 (0, 4/3, -8/9),
#                 (0, -1, 1)))
# @nb.njit(RK.class_type.instance_type(nbODEtype,
#                                       nb.float64,
#                                       nb.float64[:],
#                                       nb.float64,
#                                       nb.float64,
#                                       nbARO(1),
#                                       nbARO(1),
#                                       nb.float64),
#          cache = False) # Some issue' in making caching jitclasses
@nb.njit(cache = False)
def RK23_direct(fun: ODEFUN,
                t0: float,
                y0: npAFloat64,
                t_bound: float,
                max_step: float,
                rtol: npAFloat64,
                atol: npAFloat64,
                first_step: float) -> RK:
    return RK(fun, t0, y0, t_bound, max_step, rtol, atol, first_step,
              _RK23_error_estimator_order, _RK23_n_stages,
              _RK23_A, _RK23_B, _RK23_C, _RK23_E)
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
_RK45_error_estimator_order = np.int8(4)
_RK45_n_stages = np.int8(6)
_RK45_A = np.array((
            (0., 0., 0., 0., 0.),
            (1/5, 0., 0., 0., 0.),
            (3/40, 9/40, 0., 0., 0.),
            (44/45, -56/15, 32/9, 0., 0.),
            (19372/6561, -25360/2187, 64448/6561, -212/729, 0),
            (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656)
    ),
                    dtype = np.float64)
_RK45_B = np.array((35/384, 0, 500/1113, 125/192, -2187/6784, 11/84),
                   dtype = np.float64)
_RK45_C = np.array((0, 1/5, 3/10, 4/5, 8/9, 1), dtype = np.float64)
_RK45_E = np.array((-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40),
                   dtype = np.float64)
# @nb.njit(RK.class_type.instance_type(nbODEtype,
#                                       nb.float64,
#                                       nb.float64[:],
#                                       nb.float64,
#                                       nb.float64,
#                                       nbARO(1),
#                                       nbARO(1),
#                                       nb.float64),
#          cache = False) # Some issue in caching jitclasses
@nb.njit(cache = False)
def RK45_direct(fun: ODEFUN,
                t0: float,
                y0: npAFloat64,
                t_bound: float,
                max_step: float,
                rtol: npAFloat64,
                atol: npAFloat64,
                first_step: float) -> RK:
    return RK(fun, t0, y0, t_bound, max_step, rtol, atol, first_step,
              _RK45_error_estimator_order, _RK45_n_stages,
              _RK45_A, _RK45_B, _RK45_C, _RK45_E)
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
class Solver(enum.Enum):
    RK23 = RK23
    RK45 = RK45
    ALL = 'ALL'
# ======================================================================
# CORE FUNCTION WITH PARAMETERS AND AUXILIARY OUTPUT

def _select_initial_step_advanced(fun: ODEFUNA,
                                  t0: np.float64,
                                  y0: npAFloat64,
                                  parameters: npAFloat64,
                                  f0: npAFloat64,
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
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
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
    f1, _ = fun(t0 + h0 * direction, y1, parameters)
    d2 = norm((f1 - f0) / scale) / h0

    h1 = (max(1e-6, h0 * 1e-3) if d1 <= 1e-15 and d2 <= 1e-15
          else (max(d1, d2) * 100 ) ** error_exponent)

    return min(100 * h0, h1)
# ----------------------------------------------------------------------
def _step_advanced(fun: ODEFUNA,
                  direction: np.float64,
                  t: np.float64,
                  y: npAFloat64,
                  parameters: Any,
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
                  error_exponent: np.float64,
                  auxiliary: Any) -> tuple[bool,
                                            np.float64,
                                            npAFloat64,
                                            Any,
                                            np.float64,
                                            np.float64,
                                            npAFloat64]:
    if direction * (t - t_bound) >= 0: # t_bound has been reached
        return False, t, y, auxiliary, h_abs, h_abs, K
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
            K[s], _ = fun(t_old + C[s] * h,
                       y_old + np.dot(K[:s].T, A[s,:s]) * h,
                       parameters)

        y = y_old + h * np.dot(K[:-1].T, B)

        K[-1], auxiliary = fun(t, y, parameters)

        error_norm = norm(np.dot(K.T, E)
                          * h
                          / (atol + np.maximum(np.abs(y_old),
                                              np.abs(y)) * rtol))

        if error_norm < 1:
            h_abs *= (MAX_FACTOR if error_norm == 0 else
                            min(MAX_FACTOR,
                                SAFETY * error_norm ** error_exponent))
            return True, t, y, auxiliary, h_abs, h, K # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR,
                                SAFETY * error_norm ** error_exponent)
            if h_abs < min_step:
                return False, t, y, auxiliary, h_abs, h, K # Too small step size
# ----------------------------------------------------------------------
def Advanced(parameters_signature,
             auxiliary_signature,
             solver):

    fun_type = nbAdvanced_ODE_signature(parameters_signature,
                                        auxiliary_signature).as_type()
    nb_initial_step = nb.njit(nbAdvanced_initial_step_signature(parameters_signature,
                                                                fun_type),
                              fastmath = True)(_select_initial_step_advanced)
    nb_step_advanced = nb.njit(nbAdvanced_step_signature(parameters_signature,
                                                         auxiliary_signature,
                                                         fun_type)
                                                         )(_step_advanced)
    # ------------------------------------------------------------------
    @nb.experimental.jitclass(base_spec + (('parameters', parameters_signature),
                                           ('auxiliary', auxiliary_signature),
                                           ('fun', fun_type)))
    class RK_Advanced:
        """Base class for explicit Runge-Kutta methods."""

        def __init__(self, fun: ODEFUNA,
                     t0: float,
                     y0: npAFloat64,
                     parameters: Any,
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
            self.parameters = parameters
            self.t_bound = t_bound
            self.atol = atol
            self.rtol = rtol
            self.max_step = max_step

            self.K = np.zeros((self.n_stages + 1, len(y0)),
                              dtype = self.y.dtype)
            self.K[-1], self.auxiliary = self.fun(self.t, # type: ignore
                                                  self.y,
                                                  self.parameters)
            self.direction = np.float64(np.sign(t_bound - t0) if t_bound != t0 else 1)
            self.error_exponent = -1 / (error_estimator_order + 1)

            if not first_step:
                self.h_abs = nb_initial_step(
                    self.fun, self.t, y0, self.parameters, self.K[-1], self.direction,
                    self.error_exponent, self.rtol, self.atol)
            else:
                self.h_abs = np.abs(first_step)
            self.step_size = self.direction * self.h_abs
        # --------------------------------------------------------------
        def step(self) -> bool:
            (running,
             self.t,
             self.y,
             self.auxiliary,
             self.h_abs,
             self.step_size,
             self.K) = nb_step_advanced(self.fun,
                                        self.direction,
                                        self.t,
                                        self.y,
                                        self.parameters,
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
                                        self.error_exponent,
                                        self.auxiliary)
            return running
    # ------------------------------------------------------------------
    if solver in (Solver.RK23, Solver.ALL):
        @nb.njit(cache = False)
        def RK23_direct_advanced(fun: ODEFUNA,
                                 t0: float,
                                 y0: npAFloat64,
                                 parameters: Any,
                                 t_bound: float,
                                 max_step: float,
                                 rtol: npAFloat64,
                                 atol: npAFloat64,
                                 first_step: float) -> RK_Advanced:
            return RK_Advanced(fun, t0, y0, parameters, t_bound, max_step,
                               rtol, atol, first_step,
                     _RK23_error_estimator_order, _RK23_n_stages,
                    _RK23_A, _RK23_B, _RK23_C, _RK23_E)
        # --------------------------------------------------------------
        def RK23_advanced(fun: ODEFUNA,
                          t0: float,
                          y0: Union[int, float, npAFloat64, Iterable],
                          parameters: Any,
                          t_bound: float,
                          max_step: float = np.inf,
                          rtol: Union[int, float, npAFloat64, Iterable] = 1e-3,
                          atol: Union[int, float, npAFloat64, Iterable] = 1e-6,
                          first_step: float = 0.) -> RK_Advanced:

            y0, rtol, atol = convert(y0, rtol, atol)
            return RK23_direct_advanced(fun, t0, y0, parameters, t_bound,
                                        max_step, rtol, atol, first_step)
        # --------------------------------------------------------------
        if solver == Solver.RK23:
            return RK23_advanced
    # ------------------------------------------------------------------
    if solver in (Solver.RK45, Solver.ALL):
        @nb.njit(cache = False)
        def RK45_direct_advanced(fun,
                                 t0: float,
                                 y0: npAFloat64,
                                 parameters: Any,
                                 t_bound: float,
                                 max_step: float,
                                 rtol: npAFloat64,
                                 atol: npAFloat64,
                                 first_step: float) -> RK_Advanced:
            return RK_Advanced(fun, t0, y0, parameters, t_bound, max_step,
                               rtol, atol, first_step,
                    _RK45_error_estimator_order, _RK45_n_stages,
                    _RK45_A, _RK45_B, _RK45_C, _RK45_E)
        # --------------------------------------------------------------
        def RK45_advanced(fun: ODEFUNA,
                          t0: float,
                          y0: Union[int, float, npAFloat64, Iterable],
                          parameters: Any,
                          t_bound: float,
                          max_step: float = np.inf,
                          rtol: Union[int, float, npAFloat64, Iterable] = 1e-3,
                          atol: Union[int, float, npAFloat64, Iterable] = 1e-6,
                          first_step: float = 0.) -> RK_Advanced:

            y0, rtol, atol = convert(y0, rtol, atol)
            return RK45_direct_advanced(fun, t0, y0, parameters, t_bound,
                                        max_step, rtol, atol, first_step)
        if solver == Solver.RK45:
            return RK45_advanced
    # ------------------------------------------------------------------
    return {Solver.RK23: RK23_advanced,
            Solver.RK45: RK45_advanced}
# ======================================================================
@nb.njit # type: ignore
def step(solver) -> bool:
    return solver.step()
# ======================================================================
# FAST FORWARD
@nb.njit
def ff_to_t(solver, t_end: np.float64) -> bool:
    '''Fast forwards to given time or t_bound'''
    t_bound = solver.t_bound
    is_last = t_bound <= t_end
    if is_last:
        t_end = t_bound

    solver.t_bound = t_end

    while solver.step():
        ...

    solver.t_bound = t_bound

    return is_last
# ----------------------------------------------------------------------
@nb.njit(nb.boolean(RK.class_type.instance_type, # type: ignore
                    nb.boolean(nb.float64, nb.float64[:]).as_type()),
         cache = True)
def ff_to_cond(solver: RK, condition: Callable[[np.float64, npAFloat64], bool]
               ) -> bool:
    '''Fast forwards to given time or t_bound'''
    while solver.step():
        if condition(solver.t, solver.y):
            return True
    return False
# ----------------------------------------------------------------------
@nb.njit
def ff_to_cond_advanced(solver,
                        condition: Callable[[float, npAFloat64, Any], bool]
                        ) -> bool:
    '''Fast forwards to given time or t_bound'''
    while solver.step():
        if condition(solver.t, solver.y, solver.auxiliary):
            return True
    return False
