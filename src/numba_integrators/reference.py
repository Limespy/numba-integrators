"""Utilities for testing the package."""
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar

import numba as nb
import numba_integrators as ni
import numpy as np
import scipy
from numba_integrators._aux import nbODE2_signature
from numba_integrators._aux import nbODE_signature
from numba_integrators._aux import npAFloat64
from numba_integrators._aux import ODE2Type
from numba_integrators._aux import ODEType
# ======================================================================
# Reference initial value problems
JIT1 = nb.njit(nbODE_signature)
JIT2 = nb.njit(nbODE2_signature)
# ----------------------------------------------------------------------

@dataclass
class Problem:
    name: str
    differential: ODEType
    solution: Callable[[float], npAFloat64]
    x0: float
    x_end: float
    y0: npAFloat64 = field(init = False)
    y_end: npAFloat64 = field(init = False)
    problems: ClassVar[list[Any]] = []
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.differential = JIT1(self.differential)
        self.y0 = self.solution(self.x0)
        self.y_end = self.solution(self.x_end)
        self.problems.append(self)
# ----------------------------------------------------------------------
# Riccati
# https://en.wikipedia.org/wiki/Bernoulli_differential_equation#Example
# For initial value x = 1, y = (1, 1)
riccati = Problem('riccati',
                  lambda x, y: 2 * y/x - x ** 2 * y ** 2,
                  lambda x: np.array((x**2 / (x**5 + 4) * 5,), np.float64).T,
                  1., 20.)
# ----------------------------------------------------------------------
# Sine
# sine wave
# For initial value x = 0, y = (0, 1)
sine = Problem('sine',
               lambda x, y: np.array((y[1], -y[0])),
               lambda x: np.array((np.sin(x), np.cos(x)), np.float64).T,
               0., 10.)
# ----------------------------------------------------------------------
# Exponential
# exponential function y = exp(x)
# For initial value x = 0, y = (1)
exponential = Problem('exponential',
                      lambda x, y: y,
                      lambda x: np.array((np.exp(x),), np.float64).T,
                      0., 10.)

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
               lambda x: (np.array((np.sin(x),), np.float64).T,
                          np.array((np.cos(x),), np.float64).T),
               0., 10.)
# ----------------------------------------------------------------------
# Exponential
# exponential function y = exp(x)
# For initial value x = 0, y = (1)
exponential2 = Problem2('exponential',
                      lambda x, y, dy: y,
                      lambda x: (np.array((np.exp(x),), np.float64).T,
                                 np.array((np.exp(x),), np.float64).T),
                      0., 10.)
# ----------------------------------------------------------------------

# ======================================================================
scipy_integrators = {ni.RK23: scipy.integrate.RK23,
                     ni.RK45: scipy.integrate.RK45}
# ----------------------------------------------------------------------
def scipy_solve(solver_type,
                function: ODEType,
                x0: float,
                y0: npAFloat64,
                x_end: float,
                rtol: float | npAFloat64,
                atol: float | npAFloat64) -> npAFloat64:

    solver = scipy_integrators[solver_type](function, x0, y0, x_end,
                                            atol = atol, rtol = rtol)
    while solver.status == 'running':
        solver.step()
    assert solver.t == x_end
    return solver.y
