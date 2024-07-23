"""Utilities for testing the package."""
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar

import numba as nb
import numpy as np
from numpy import cos
from numpy import exp
from numpy import sin

from ..._aux import npAFloat64
from ._second_basic_aux import nbODE2_signature
from ._second_basic_aux import ODE2Type
# ======================================================================
# Reference initial value problems
JIT2 = nb.njit(nbODE2_signature)
# ----------------------------------------------------------------------
def arr(*iterable):
    return np.array(iterable, np.float64).T
# ======================================================================
@dataclass
class Problem2:
    name: str
    differential: ODE2Type
    solution: Callable[[float], tuple[npAFloat64, npAFloat64]]
    x0: float
    x_end: float
    y0: npAFloat64 = field(init = False)
    dy0: npAFloat64 = field(init = False)
    y_end: npAFloat64 = field(init = False)
    dy_end: npAFloat64 = field(init = False)
    problems: ClassVar[list[Any]] = []
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.differential = JIT2(self.differential)
        self.y0, self.dy0 = self.solution(self.x0)
        self.y_end, self.dy_end = self.solution(self.x_end)
        self.problems.append(self)
# ----------------------------------------------------------------------
# Sine
# sine wave
# For initial value x = 0, y = (0, 1)
sine2 = Problem2('sine',
               lambda x, y, dy: -y,
               lambda x: (arr(np.sin(x),),
                          arr(np.cos(x),)),
               0., 10.)
# ----------------------------------------------------------------------
# Exponential
# exponential function y = exp(x)
# For initial value x = 0, y = (1)
exponential2 = Problem2('exponential',
                      lambda x, y, dy: y,
                      lambda x: (arr(exp(x),),
                                 arr(exp(x),)),
                      0., 10.)
# ----------------------------------------------------------------------
# Mass-spring
# https://kyleniemeyer.github.io/ME373-book/content/second-order/numerical-methods.html
mass_spring2 = Problem2('mass_spring',
                       lambda x, y, dy: 10 * np.sin(x) - 5. * dy - 6. * y,
    lambda x: (arr(-6. * exp(-3. * x) + 7. * exp(-2. * x) + sin(x) - cos(x)),
               arr(18. * exp(-3. * x) - 14. * exp(-2. * x) + cos(x) + sin(x))),
               0., 3.)
# ----------------------------------------------------------------------
# Exp SIn
# y(x) = exp(sin(x))
# y'(x) = cos(x) * exp(sin(x)) = cos(x) * y(x)
# y''(x) = -sin(x) * y(x) + cos(x) * y'(x)
exp_sin2 = Problem2('exp_sin',
                       lambda x, y, dy: cos(x) * dy - sin(x) * y,
    lambda x: (arr(exp(sin(x))),
               arr(cos(x) * exp(sin(x)))),
               0., 3.)
