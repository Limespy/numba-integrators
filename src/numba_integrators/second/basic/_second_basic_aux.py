from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from ..._aux import calc_tolerance
from ..._aux import convert
from ..._aux import nbDecFC
from ..._aux import norm
from ..._aux import SolverBase
# ----------------------------------------------------------------------
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from ..._aux import npAFloat64
    from ..._aux import Arrayable
    from ..._aux import SolverType

    ODE2Type: TypeAlias  = Callable[[np.float64, npAFloat64, npAFloat64],
                                    npAFloat64]

    _InitType: TypeAlias = Callable[[ODE2Type,
                                    float,
                                    Arrayable,
                                    Arrayable,
                                    float,
                                    float,
                                    Arrayable,
                                    Arrayable,
                                    float], SolverType]
else:
    Arrayable = SolverType = ODE2Type = _InitType = npAFloat64 = None
# ======================================================================
nbODE2_signature = nb.float64[:](nb.float64, nb.float64[:], nb.float64[:])
nbODE2_type = nbODE2_signature.as_type()
# ======================================================================
class Solver2:
    _solver_params: tuple
    _init: _InitType
    # ------------------------------------------------------------------
    def __new__(cls, # type: ignore[misc]
                fun: ODE2Type,
                x0: float,
                y0: Arrayable,
                dy0: Arrayable,
                x_bound: float,
                *,
                max_step: float = np.inf,
                rtol: Arrayable = 1e-3,
                atol: Arrayable = 1e-6,
                first_step: float = 0.) -> SolverType:
        y0, dy0, rtol, atol = convert(y0, dy0, rtol, atol)
        return cls._init(fun,
                        np.float64(x0),
                        y0,
                        dy0,
                        np.float64(x_bound),
                        np.float64(max_step),
                        rtol,
                        atol,
                        np.float64(first_step),
                        tuple(cls._solver_params))
# ======================================================================
@nbDecFC
def calc_h0(y0: npAFloat64,
            dy0: npAFloat64,
            ddy0: npAFloat64,
            direction: np.float64,
            scale: npAFloat64):
    d0 = norm(dy0 / scale)
    d1 = norm(ddy0 / scale)

    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    Dx = h0 * direction
    Ddy = Dx * ddy0
    dy1 = dy0 + Ddy
    y1 = y0 + dy0 * Dx + Ddy * 0.5 * Dx
    return h0, y1, dy1, d1
# ----------------------------------------------------------------------
@nbDecFC
def calc_h_abs(y_diff: npAFloat64,
               h0: np.float64,
               scale: npAFloat64,
               error_exponent: np.float64,
               d1: np.float64):
    d2 = norm(y_diff / scale) / h0

    return min(100. * h0,
               (max(1e-6, h0 * 1e-3) if d1 <= 1e-15 and d2 <= 1e-15
                else (max(d1, d2) * 100.) ** (2. *error_exponent)))
# ----------------------------------------------------------------------
# @nbDec(nb.float64(nbODE2_type,
#                     nb.float64,
#                     nbA(1),
#                     nbA(1),
#                     nbA(1),
#                     nb.float64,
#                     nb.float64,
#                     nbARO(1),
#                     nbARO(1)),
#          fastmath = True, cache = IS_CACHE)
@nbDecFC
def select_initial_step(fun: ODE2Type,
                        x0: np.float64,
                        y0: npAFloat64,
                        dy0: npAFloat64,
                        ddy0: npAFloat64,
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
    scale = calc_tolerance(np.abs(dy0), rtol, atol)
    h0, y1, dy1, d1 = calc_h0(y0, dy0, ddy0, direction, scale)
    ddy1 = fun(x0 + h0 * direction, y1, dy1)
    return calc_h_abs(ddy1 - ddy0, h0, scale, error_exponent, d1)
# ======================================================================
class SecondBasicSolverBase(SolverBase):
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[np.float64, npAFloat64, npAFloat64, npAFloat64]:
        return self.x, self.y, self.dy, self.ddy
