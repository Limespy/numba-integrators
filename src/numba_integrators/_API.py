import enum
from typing import Any
from typing import Callable
from typing import Union

import numba as nb
import numpy as np

from ._aux import Float64Array
from ._aux import nbA
from ._aux import nbAdvanced_initial_step_signature
from ._aux import nbAdvanced_ODE_signature
from ._aux import nbAdvanced_step_signature
from ._aux import nbARO
from ._aux import nbODEtype
from ._aux import nbtype

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

@nb.njit(nb.float64(nb.float64[:]),
         fastmath = True, cache = True)
def norm(x):
    """Compute RMS norm."""
    return np.sqrt(np.sum(x * x) / x.size)

@nb.njit(nb.float64(nbODEtype,
                    nb.float64,
                    nb.float64[:],
                    nb.float64[:],
                    nb.int8,
                    nb.int8,
                    nbARO(1),
                    nbARO(1)),
         fastmath = True, cache = True)
def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
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
    if y0.size == 0:
        return np.inf

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)

    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    h1 = (max(1e-6, h0 * 1e-3) if d1 <= 1e-15 and d2 <= 1e-15
          else (0.01 / max(d1, d2)) ** (1 / (order + 1)))

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
        cache = True)
def _step(fun, direction, t, y, t_bound, h_abs, max_step, K, n_stages,
          rtol, atol, A, B, C, E, error_exponent):
    if direction * (t - t_bound) >= 0:
        return False, t, y, h_abs, h_abs, K # t_bound has been reached
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
             ('order', nb.int8),
             ('error_estimator_order', nb.int8),
             ('n_stages', nb.int8),
             ('t', nb.float64),
             ('y', nb.float64[:]),
             ('t_bound', nb.float64),
             ('direction', nb.int8),
             ('max_step', nb.float64),
             ('error_exponent', nb.float64),
             ('h_abs', nb.float64),
             ('step_size', nb.float64),
             ('atol', nbARO(1)),
             ('rtol', nbARO(1))
)

@nb.experimental.jitclass(base_spec + (('fun', nbODEtype),))
class RK:
    """Base class for explicit Runge-Kutta methods."""

    def __init__(self, fun, t0, y0, t_bound, max_step, rtol, atol, first_step,
                 order, error_estimator_order, n_stages, A, B, C, E):
        self.order = order
        self.error_estimator_order = error_estimator_order
        self.n_stages = n_stages
        self.A = A
        self.B = B
        self.C = C
        self.E = E
        self.fun = fun
        self.t = t0
        self.y = y0
        self.t_bound = t_bound
        self.K = np.zeros((self.n_stages + 1, len(y0)), dtype = self.y.dtype)
        self.K[-1] = self.fun(self.t, self.y)
        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.t, y0, self.K[-1], self.direction,
                self.error_estimator_order, self.rtol, self.atol)
        else:
            self.h_abs = first_step
        self.step_size = self.h_abs
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
def convert(y0, rtol, atol) -> tuple[Float64Array, Float64Array, Float64Array]:
    y0 = np.asarray(y0).astype(np.float64)
    if y0.ndim == 0:
        y0 = np.full(1, y0)

    if not isinstance(rtol, np.ndarray) or len(rtol) == 1:
        rtol = np.full(len(y0), rtol)

    if not isinstance(atol, np.ndarray) or len(atol) == 1:
        atol = np.full(len(y0), atol)

    return y0, rtol, atol
# ======================================================================
_RK23_order = 3
_RK23_error_estimator_order = 2
_RK23_n_stages = 3
_RK23_A = np.array((
    (0, 0, 0),
    (1/2, 0, 0),
    (0, 3/4, 0)
))
_RK23_B = np.array((2/9, 1/3, 4/9))
_RK23_C = np.array((0, 1/2, 3/4))
_RK23_E = np.array((5/72, -1/12, -1/9, 1/8))
_RK23_P = np.array(((1, -4 / 3, 5 / 9),
                (0, 1, -2/3),
                (0, 4/3, -8/9),
                (0, -1, 1)))
# @nb.njit(RK.class_type.'instance_type(nbODEtype,
#                                       nb.float64,
#                                       nb.float64[:],
#                                       nb.float64,
#                                       nb.float64,
#                                       nbARO(1),
#                                       nbARO(1),
#                                       nb.float64),
#          cache = False) # Some issue' in making caching jitclasses
@nb.njit(cache = False)
def RK23_direct(fun, t0, y0, t_bound, max_step, rtol, atol, first_step):
    return RK(fun, t0, y0, t_bound, max_step, rtol, atol, first_step,
              _RK23_order, _RK23_error_estimator_order, _RK23_n_stages,
              _RK23_A, _RK23_B, _RK23_C, _RK23_E)
# ----------------------------------------------------------------------
def RK23(fun, t0, y0, t_bound, max_step = np.inf,
         rtol = 1e-3, atol = 1e-6, first_step = 0.):

    y0, rtol, atol = convert(y0, rtol, atol)
    return RK23_direct(fun, t0, y0, t_bound, max_step, rtol, atol, first_step)
# ----------------------------------------------------------------------
_RK45_order = np.int8(5)
_RK45_error_estimator_order = np.int8(4)
_RK45_n_stages = np.int8(6)
_RK45_A = np.array((
            (0., 0., 0., 0., 0.),
            (1/5, 0., 0., 0., 0.),
            (3/40, 9/40, 0., 0., 0.),
            (44/45, -56/15, 32/9, 0., 0.),
            (19372/6561, -25360/2187, 64448/6561, -212/729, 0),
            (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656)
    ), dtype = np.float64)
_RK45_B = np.array((35/384, 0, 500/1113, 125/192, -2187/6784, 11/84))
_RK45_C = np.array((0, 1/5, 3/10, 4/5, 8/9, 1))
_RK45_E = np.array((-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40))
# @nb.njit(RK.class_type.instance_type(nbODEtype,
#                                       nb.float64,
#                                       nb.float64[:],
#                                       nb.float64,
#                                       nb.float64,
#                                       nbARO(1),
#                                       nbARO(1),
#                                       nb.float64),
#          cache = False) # Some isse in making caching jitclasses
@nb.njit(cache = False)
def RK45_direct(fun, t0, y0, t_bound, max_step, rtol, atol, first_step):
    return RK(fun, t0, y0, t_bound, max_step, rtol, atol, first_step,
              _RK45_order, _RK45_error_estimator_order, _RK45_n_stages,
              _RK45_A, _RK45_B, _RK45_C, _RK45_E)
# ----------------------------------------------------------------------
def RK45(fun, t0, y0, t_bound, max_step = np.inf,
         rtol = 1e-3, atol = 1e-6, first_step = 0):

    y0, rtol, atol = convert(y0, rtol, atol)
    return RK45_direct(fun, t0, y0, t_bound, max_step, rtol, atol, first_step)
# ----------------------------------------------------------------------
class Solver(enum.Enum):
    RK23 = RK23
    RK45 = RK45
    ALL = 'ALL'
# ======================================================================
# CORE FUNCTION WITH PARAMETERS AND AUXILIARY OUTPUT

def _select_initial_step_advanced(fun, t0, y0, parameters, f0, direction,
                                  order, rtol, atol) -> float:
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
    if y0.size == 0:
        return np.inf

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)

    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1, _ = fun(t0 + h0 * direction, y1, parameters)
    d2 = norm((f1 - f0) / scale) / h0

    h1 = (max(1e-6, h0 * 1e-3) if d1 <= 1e-15 and d2 <= 1e-15
          else (0.01 / max(d1, d2)) ** (1 / (order + 1)))

    return min(100 * h0, h1)
# ----------------------------------------------------------------------
def _step_advanced(fun, direction, t, y, parameters, t_bound, h_abs, max_step,
                   K, n_stages, rtol, atol, A, B, C, E, error_exponent,
                   auxiliary):
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

        def __init__(self, fun, t0, y0, parameters, t_bound, max_step, rtol, atol, first_step,
                    order, error_estimator_order, n_stages, A, B, C, E):
            self.order = order
            self.error_estimator_order = error_estimator_order
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
            self.K = np.zeros((self.n_stages + 1, len(y0)), dtype = self.y.dtype)
            self.K[-1], self.auxiliary = self.fun(self.t, self.y, self.parameters)
            self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
            self.error_exponent = -1 / (self.error_estimator_order + 1)
            self.atol = atol
            self.rtol = rtol
            self.max_step = max_step

            if not first_step:
                self.h_abs = nb_initial_step(
                    self.fun, self.t, y0, self.parameters, self.K[-1], self.direction,
                    self.error_estimator_order, self.rtol, self.atol)
            else:
                self.h_abs = first_step
            self.step_size = self.h_abs
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
        def RK23_direct_advanced(fun, t0, y0, parameters, t_bound, max_step,
                                 rtol, atol, first_step):
            return RK_Advanced(fun, t0, y0, parameters, t_bound, max_step,
                               rtol, atol, first_step,
                    _RK23_order, _RK23_error_estimator_order, _RK23_n_stages,
                    _RK23_A, _RK23_B, _RK23_C, _RK23_E)
        # --------------------------------------------------------------
        def RK23_advanced(fun, t0, y0, parameters, t_bound, max_step = np.inf,
                rtol = 1e-3, atol = 1e-6, first_step = 0.):

            y0, rtol, atol = convert(y0, rtol, atol)
            return RK23_direct_advanced(fun, t0, y0, parameters, t_bound,
                                        max_step, rtol, atol, first_step)
        # --------------------------------------------------------------
        if solver == Solver.RK23:
            return RK23_advanced
    # ------------------------------------------------------------------
    if solver in (Solver.RK45, Solver.ALL):
        @nb.njit(cache = False)
        def RK45_direct_advanced(fun, t0, y0, parameters, t_bound, max_step,
                                 rtol, atol, first_step):
            return RK_Advanced(fun, t0, y0, parameters, t_bound, max_step,
                               rtol, atol, first_step,
                    _RK45_order, _RK45_error_estimator_order, _RK45_n_stages,
                    _RK45_A, _RK45_B, _RK45_C, _RK45_E)
        # --------------------------------------------------------------
        def RK45_advanced(fun, t0, y0, parameters, t_bound, max_step = np.inf,
                rtol = 1e-3, atol = 1e-6, first_step = 0):

            y0, rtol, atol = convert(y0, rtol, atol)
            return RK45_direct_advanced(fun, t0, y0, parameters, t_bound,
                                        max_step, rtol, atol, first_step)
        if solver == Solver.RK45:
            return RK45_advanced

    return {Solver.RK23: RK23_advanced,
            Solver.RK45: RK45_advanced}
# ======================================================================
@nb.njit # type: ignore
def step(solver: RK) -> bool:
    return solver.step()
