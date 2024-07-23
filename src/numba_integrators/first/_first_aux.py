from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from .._aux import convert
from .._aux import nbDecFC
from .._aux import norm
from .._aux import SolverBase
# ----------------------------------------------------------------------
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from .._aux import npAFloat64
    from .._aux import Arrayable
    from .._aux import SolverType

    ODE1Type: TypeAlias  = Callable[[np.float64, npAFloat64], npAFloat64]
    _InitType: TypeAlias = Callable[[ODE1Type,
                                        float,
                                        Arrayable,
                                        Arrayable,
                                        float,
                                        float,
                                        Arrayable,
                                        Arrayable,
                                        float], SolverType]
else:
    npAFloat64 = Arrayable = SolverType = ODE1Type = _InitType = None
# ======================================================================
nbODE_signature = nb.float64[:](nb.float64, nb.float64[:])
nbODE_type = nbODE_signature.as_type()
# ======================================================================
@nbDecFC
def calc_h0(y0: npAFloat64,
            dy0: npAFloat64,
            direction: np.float64,
            scale: npAFloat64):
    d0 = norm(y0 / scale)
    d1 = norm(dy0 / scale)

    h_abs = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    h = np.copysign(h_abs, direction)
    y1 = y0 + h * dy0
    return h, h_abs, y1, d1
# ----------------------------------------------------------------------
@nbDecFC
def calc_h_abs(dy_diff: npAFloat64,
               h_abs: np.float64,
               scale: npAFloat64,
               error_exponent: np.float64,
               d1: np.float64):
    d2 = norm(dy_diff / scale) / h_abs

    return min(100. * h_abs,
               (max(1e-6, h_abs * 1e-3) if d1 <= 1e-15 and d2 <= 1e-15
                else (max(d1, d2) * 100.) **(2.* error_exponent)))
# ======================================================================
class Solver1:
    _solver_params: tuple
    _init: _InitType
    # ------------------------------------------------------------------
    def __new__(cls, # type: ignore[misc]
                fun: ODE1Type,
                x0: float,
                y0: Arrayable,
                x_bound: float,
                *,
                max_step: float = np.inf,
                rtol: Arrayable = 1e-3,
                atol: Arrayable = 1e-6,
                first_step: float = 0.) -> SolverType:
        y0, rtol, atol = convert(y0, rtol, atol)
        return cls._init(fun,
                        np.float64(x0),
                        y0,
                        np.float64(x_bound),
                        np.float64(max_step),
                        rtol,
                        atol,
                        np.float64(first_step),
                        tuple(cls._solver_params))
# ======================================================================
class FirstSolverBase(SolverBase):
    # ------------------------------------------------------------------
    @property
    def dy(self) -> npAFloat64:
        return self.K[-1]
