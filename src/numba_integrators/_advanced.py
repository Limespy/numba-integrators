from typing import Any

import numba as nb
import numpy as np

from ._aux import Arrayable
from ._aux import convert
from ._aux import MAX_FACTOR
from ._aux import MIN_FACTOR
from ._aux import nbA
from ._aux import nbARO
from ._aux import norm
from ._aux import npAFloat64
from ._aux import ODEFUNA
from ._aux import RK23_params
from ._aux import RK45_params
from ._aux import SAFETY
from ._aux import Solver
from ._basic import base_spec
from ._basic import Solvers
# ----------------------------------------------------------------------
def nbAdvanced_ODE_signature(parameters_type, auxiliary_type):
    return nb.types.Tuple((nb.float64[:],
                           auxiliary_type))(nb.float64,
                                            nb.float64[:],
                                            parameters_type)
# ----------------------------------------------------------------------
def nbAdvanced_initial_step_signature(parameters_type, fun_type):
    return nb.float64(fun_type,
                        nb.float64,
                        nb.float64[:],
                        parameters_type,
                        nb.float64[:],
                        nb.float64,
                        nb.float64,
                        nbARO(1),
                        nbARO(1))
# ======================================================================
def select_initialstep_advanced(fun: ODEFUNA,
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
def nbAdvanced_step_signature(parameters_type,
                              auxiliary_type,
                              fun_type):
    return nb.types.Tuple((nb.boolean,
                           nb.float64,
                           nb.float64[:],
                           auxiliary_type,
                           nb.float64,
                           nb.float64,
                           nbA(2)))(fun_type,
                                    nb.float64,
                                    nb.float64,
                                    nb.float64[:],
                                    parameters_type,
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
                                    nb.float64,
                                    auxiliary_type)
# ----------------------------------------------------------------------
def step_advanced(fun: ODEFUNA,
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
    eps = np.abs(np.nextafter(t_old, direction * np.inf) - t_old)
    min_step = 8 * eps

    if h_abs < min_step:
        h_abs = min_step

    while True: # := not working
        if h_abs > max_step:
            h_abs = max_step - eps
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
                      min(MAX_FACTOR, SAFETY * error_norm ** error_exponent))
            return True, t, y, auxiliary, h_abs, h, K # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** error_exponent)
            if h_abs < min_step:
                return False, t, y, auxiliary, h_abs, h, K # Too small step size
# ----------------------------------------------------------------------
class _RK_Advanced(Solver):
        """Base class for advanced version of explicit Runge-Kutta methods."""

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
                     E: npAFloat64,
                     nb_initial_step,
                     nbstep_advanced):
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
            self.initial_step = nb_initial_step
            self._step = nbstep_advanced

            self.K = np.zeros((self.n_stages + 1, len(y0)),
                              dtype = self.y.dtype)
            self.K[-1], self.auxiliary = self.fun(self.t, # type: ignore
                                                  self.y,
                                                  self.parameters)
            self.direction = np.float64(np.sign(t_bound - t0) if t_bound != t0 else 1)
            self.error_exponent = -1 / (error_estimator_order + 1)

            if not first_step:
                self.h_abs = self.initial_step(
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
             self.K) = self._step(self.fun,
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
# ----------------------------------------------------------------------
def Advanced(parameters_signature,
             auxiliary_signature,
             solver: Solvers):

    fun_type = nbAdvanced_ODE_signature(parameters_signature,
                                        auxiliary_signature).as_type()
    signature_initial_step = nbAdvanced_initial_step_signature(
        parameters_signature, fun_type)
    nb_initial_step = nb.njit(signature_initial_step,
                              fastmath = True)(select_initialstep_advanced)
    signature_step = nbAdvanced_step_signature(parameters_signature,
                                               auxiliary_signature,
                                               fun_type)
    nbstep_advanced = nb.njit(signature_step)(step_advanced)
    # ------------------------------------------------------------------

    RK_Advanced = nb.experimental.jitclass(
        base_spec + (('parameters', parameters_signature),
                     ('auxiliary', auxiliary_signature),
                     ('fun', fun_type),
                     ('initial_step', signature_initial_step.as_type()),
                     ('_step', signature_step.as_type()))
        )(_RK_Advanced)
    # ------------------------------------------------------------------
    if solver in (Solvers.RK23, Solvers.ALL):
        @nb.njit
        def RK23_direct_advanced(fun: ODEFUNA,
                                 t0: float,
                                 y0: npAFloat64,
                                 parameters: Any,
                                 t_bound: float,
                                 max_step: float,
                                 rtol: npAFloat64,
                                 atol: npAFloat64,
                                 first_step: float) -> _RK_Advanced:
            return RK_Advanced(fun, t0, y0, parameters, t_bound, max_step,
                               rtol, atol, first_step, *RK23_params,
                               nb_initial_step, nbstep_advanced)
        # --------------------------------------------------------------
        def RK23_advanced(fun: ODEFUNA,
                          t0: float,
                          y0: Arrayable,
                          parameters: Any,
                          t_bound: float,
                          max_step: float = np.inf,
                          rtol: Arrayable = 1e-3,
                          atol: Arrayable = 1e-6,
                          first_step: float = 0.) -> _RK_Advanced:

            y0, rtol, atol = convert(y0, rtol, atol)
            return RK23_direct_advanced(fun, t0, y0, parameters, t_bound,
                                        max_step, rtol, atol, first_step)
        # --------------------------------------------------------------
        if solver == Solvers.RK23:
            return RK23_advanced
    # ------------------------------------------------------------------
    if solver in (Solvers.RK45, Solvers.ALL):
        @nb.njit
        def RK45_direct_advanced(fun,
                                 t0: float,
                                 y0: npAFloat64,
                                 parameters: Any,
                                 t_bound: float,
                                 max_step: float,
                                 rtol: npAFloat64,
                                 atol: npAFloat64,
                                 first_step: float) -> _RK_Advanced:
            return RK_Advanced(fun, t0, y0, parameters, t_bound, max_step,
                               rtol, atol, first_step, *RK45_params,
                               nb_initial_step, nbstep_advanced)
        # --------------------------------------------------------------
        def RK45_advanced(fun: ODEFUNA,
                          t0: float,
                          y0: Arrayable,
                          parameters: Any,
                          t_bound: float,
                          max_step: float = np.inf,
                          rtol: Arrayable = 1e-3,
                          atol: Arrayable = 1e-6,
                          first_step: float = 0.) -> _RK_Advanced:

            y0, rtol, atol = convert(y0, rtol, atol)
            return RK45_direct_advanced(fun, t0, y0, parameters, t_bound,
                                        max_step, rtol, atol, first_step)
        if solver == Solvers.RK45:
            return RK45_advanced
    # ------------------------------------------------------------------
    return {Solvers.RK23: RK23_advanced,
            Solvers.RK45: RK45_advanced}
