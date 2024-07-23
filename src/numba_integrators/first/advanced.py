from collections.abc import Callable
from typing import Any
from typing import TypeAlias

import numba as nb
import numpy as np

from .._aux import _ODEA_return
from .._aux import Arrayable
from .._aux import calc_eps
from .._aux import calc_error
from .._aux import calc_tolerance
from .._aux import convert
from .._aux import FirstSolverBase
from .._aux import jitclass_from_dict
from .._aux import MAX_FACTOR
from .._aux import MIN_FACTOR
from .._aux import nbA
from .._aux import nbARO
from .._aux import nbSignature
from .._aux import nbType
from .._aux import npAFloat64
from .._aux import RK_Params
from .._aux import SAFETY
from .._aux import Solver
from .._aux import step_prep
from ._first_aux import calc_h0
from ._first_aux import calc_h_abs
# ======================================================================
ODEAType: TypeAlias = Callable[[np.float64, npAFloat64, Any],
                              _ODEA_return]
# ======================================================================
def nbAdvanced_ODE_signature(parameters_type: nbType,
                             auxiliary_type: nbType) -> nbSignature:
    return nb.types.Tuple((nb.float64[:],
                           auxiliary_type))(nb.float64,
                                            nb.float64[:],
                                            parameters_type)
# ----------------------------------------------------------------------
def nbAdvanced_initial_step_signature(parameters_type: nbType,
                                      fun_type: nbType) -> nbSignature:
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
InitialStepFType: TypeAlias = Callable[[ODEAType,
                                        np.float64,
                                        npAFloat64,
                                        Any,
                                        npAFloat64,
                                        np.float64,
                                        np.float64,
                                        npAFloat64,
                                        npAFloat64], np.float64]
def select_initial_step(fun: ODEAType,
                        x0: np.float64,
                        y0: npAFloat64,
                        parameters: Any,
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
    x0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(x0, y0)``.
    direction : float
        Integration direction.
    error_exponent : np.float64
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

    scale = calc_tolerance(np.abs(y0), rtol, atol)
    h0, y1, d1 = calc_h0(y0, dy0, direction, scale)
    dy1, _ = fun(x0 + h0 * direction, y1, parameters)[0]
    return calc_h_abs(dy1 - dy0, h0, scale, error_exponent, d1)
# ----------------------------------------------------------------------
def nbAdvanced_step_signature(parameters_type: nbType,
                              auxiliary_type: nbType,
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
def step(fun: ODEAType,
        direction: np.float64,
        x0: np.float64,
        y0: npAFloat64,
        parameters: Any,
        h_abs: np.float64,
        h: np.float64,
        min_step: np.float64,
        K: npAFloat64,
        rtol: npAFloat64,
        atol: npAFloat64,
        _len: np.float64,
        n_stages: np.int64,
        A: npAFloat64,
        C: npAFloat64,
        error_exponent: np.float64
        ) -> tuple[bool,
                   np.float64,
                   npAFloat64,
                   Any,
                   np.float64,
                   np.float64,
                   np.int64]:
    nfev = np.int64(0)
    y0_abs = np.abs(y0)
    while True: # := not working
        h = direction * h_abs

        # RK core loop
        Dx = C[0] * h
        K[1], _ = fun(x0 + Dx, y0 + Dx * K[0], parameters)

        for s in range(2, n_stages):
            Dx = C[s - 1] * h
            K[s], _ = fun(x0 + Dx,
                          y0 + np.dot(A[s - 2,:s] * Dx, K[:s]),
                          parameters)

        # Last step
        x = x0 + h
        y = y0 + np.dot(A[-2,:-1] * h, K[:-1])
        K[-1], auxiliary = fun(x, y, parameters)

        nfev += n_stages

        error = calc_error(np.dot(A[-1], K), y, y0_abs, rtol, atol
                           ) * _len * h * h

        if error < 1.:
            h_abs *= (MAX_FACTOR if error == 0 else
                      min(MAX_FACTOR, SAFETY * error ** error_exponent))
            if h_abs < min_step: # Due to the SAFETY, the step can shrink
                h_abs = min_step
            return True, x, y, auxiliary, h_abs, h, nfev # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error ** error_exponent)
            if h_abs < min_step:
                return False, x, y, auxiliary, h_abs, h, nfev # Too small step size
# ----------------------------------------------------------------------
class RKA(FirstSolverBase):
    """Base class for advanced version of explicit Runge-Kutta methods."""

    def __init__(self, fun: ODEAType,
                    x0: np.float64,
                    y0: npAFloat64,
                    parameters: Any,
                    x_bound: np.float64,
                    max_step: np.float64,
                    rtol: npAFloat64,
                    atol: npAFloat64,
                    first_step: np.float64,
                    n_stages: np.int64,
                    A: npAFloat64,
                    C: npAFloat64,
                    error_exponent: np.float64,
                    nb_initial_step: InitialStepFType,
                    nbstep_advanced):
        self.fun = fun
        self.x = x0
        self.y = y0
        self.parameter = parameters
        self.x_bound = x_bound
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.n_stages = n_stages
        self.A = A
        self.C = C
        self.error_exponent = error_exponent
        self.initial_step = nb_initial_step
        self._A_step = nbstep_advanced

        self.reboot(first_step)
    # ------------------------------------------------------------------
    def _init_K(self, y_len: int):
        self.K = np.zeros((self.n_stages + np.int64(1), y_len), np.float64)
        self.K[0], self.auxiliary = self.fun(self.x, self.y, self.parameters)
    # ------------------------------------------------------------------
    def _select_initial_step(self) -> np.float64:
        return self.initial_step(self.fun, self.x, self.y, self.parameters,
                                 self.K[-1], self.direction,
                                 self.error_exponent, self.rtol, self.atol)
    # --------------------------------------------------------------
    def _step(self, h_abs: np.float64, h_abs_min: np.float64) -> bool:
        (valid,
        self.x,
        self.y,
        self.auxiliary,
        self.h_abs,
        self.step_size,
        nfev) = self._A_step(self.fun,
                            self.direction,
                            self.x,
                            self.y,
                            self.parameters,
                            h_abs,
                            self.step_size,
                            h_abs_min,
                            self.K,
                            self.rtol,
                            self.atol,
                            self._len,
                            self.n_stages,
                            self.A,
                            self.C,
                            self.error_exponent)
        self._nfev += nfev
        self.K[0] = self.K[-1]
        return valid
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[np.float64, npAFloat64, npAFloat64, Any]:
        return self.x, self.y, self.K[0], self.auxiliary
# ----------------------------------------------------------------------
AdvancedSolver: TypeAlias = Callable[[ODEAType,
                                      float,
                                      Arrayable,
                                      Any,
                                      float,
                                      float,
                                      Arrayable,
                                      Arrayable,
                                      float],
                                     RKA]
# ----------------------------------------------------------------------
def _Advanced_make_init_RK(parameters_type: nbType,
                           auxiliary_type: nbType):

    fun_type = nbAdvanced_ODE_signature(parameters_type,
                                        auxiliary_type).as_type()
    signature_initial_step = nbAdvanced_initial_step_signature(
        parameters_type, fun_type)
    nb_initial_step = nb.njit(signature_initial_step,
                              fastmath = True)(select_initial_step)
    signature_step = nbAdvanced_step_signature(parameters_type,
                                               auxiliary_type,
                                               fun_type)
    nbstep_advanced = nb.njit(signature_step)(step)

    nbRKA = jitclass_from_dict(
                  {'parameters': parameters_type,
                     'auxiliary': auxiliary_type,
                     'fun': fun_type,
                     'initial_step': signature_initial_step.as_type(),
                     '_step': signature_step.as_type()}
                                           )(RKA)
    # ------------------------------------------------------------------
    @nb.njit
    def init_RK(fun: ODEAType,
            x0: np.float64,
            y0: npAFloat64,
            parameters,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params: RK_Params) -> RKA:
        return nbRKA(fun,
                x0,
                y0,
                parameters,
                x_bound,
                max_step,
                rtol,
                atol,
                first_step,
                *solver_params,
                nb_initial_step,
                nbstep_advanced)
    return init_RK
# ----------------------------------------------------------------------
def _Advanced_nb_prep(parameters_type: nbType,
                      auxiliary_type: nbType,
                      signature_hash: int):
    if (output := _advanced_solver_cache.get(signature_hash)) is None:
        output = _Advanced_make_init_RK(parameters_type,
                                        auxiliary_type)
        _advanced_solver_cache[signature_hash] = output
    return output
# ----------------------------------------------------------------------
def _Advanced(parameters_type: nbType,
             auxiliary_type: nbType,
             solver: Solver,
             signature_hash: int) -> AdvancedSolver:
    """Generates the advanced solver based on the given signatures."""
    solver_params = solver._solver_params
    init_RK = _Advanced_nb_prep(parameters_type,
                                auxiliary_type,
                                signature_hash)

    def RK_advanced(fun: ODEAType,
                    x0: float,
                    y0: Arrayable,
                    parameters: Any,
                    x_bound: float,
                    *,
                    max_step: float = np.inf,
                    rtol: Arrayable = 1e-6,
                    atol: Arrayable = 1e-6,
                    first_step: float = 0.) -> RKA:
        y0, rtol, atol = convert(y0, rtol, atol)
        return init_RK(fun,
                        np.float64(x0),
                        y0,
                        parameters,
                        np.float64(x_bound),
                        np.float64(max_step),
                        rtol,
                        atol,
                        np.float64(first_step),
                        solver_params)
    # ------------------------------------------------------------------
    return RK_advanced # type: ignore[return-value]
# ----------------------------------------------------------------------
_advanced_solver_cache: dict[int, AdvancedSolver] = {}
# ----------------------------------------------------------------------
def Advanced(parameters_type: nbType,
             auxiliary_type: nbType,
             solver: Solver) -> AdvancedSolver:
    """Generates the advanced solver based on the given signatures.

    Uses memoization
    """
    signature_hash = hash(parameters_type) ^ hash(auxiliary_type)
    full_hash = signature_hash ^ hash(solver)
    if (output := _advanced_solver_cache.get(full_hash)) is None:
        output = _Advanced(parameters_type,
                           auxiliary_type, solver,
                           signature_hash)
        _advanced_solver_cache[full_hash] = output
    return output
