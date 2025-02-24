from typing import Literal as L
from typing import TypeAlias

import numpy as np
from numba_integrators._lnumpy import F64Array
# ======================================================================
Y1: TypeAlias = F64Array[L[1]]
# ======================================================================
def diffs0(x: float) -> tuple[Y1, Y1, Y1, Y1]:
    x = np.array((x,))
    y = np.sin(x)
    dy = np.cos(x)
    return (y, dy, -y, -dy)
# ======================================================================
def ddy0(x: float, y: Y1, dy: Y1) -> Y1:
    return -y
# ======================================================================
def jac0(x: float, y: Y1, dy: Y1
         ) -> F64Array[L[1], L[2]]:
    return np.array(((-1., 0.),))
# ======================================================================
def diffs1(x: float) -> tuple[Y1, Y1,Y1, Y1]:
    x = np.array((x,))
    _sin = np.sin(x)
    _cos = np.cos(x)
    y = 50. * (_sin + 50. * _cos - 50. * np.exp(-50. * x)) / 2501.
    dy = 50. * (_cos - y)
    ddy = 50. * (- _sin - dy)
    dddy = 50. * (- _cos - ddy)
    return (y, dy, ddy, dddy)
# ======================================================================
def ddy1(x: float, y: Y1, dy: Y1) -> Y1:
    return 50. * (- np.sin(x) - dy)
# ======================================================================
def jac1(x: float, y: Y1, dy: Y1
         ) -> F64Array[L[1], L[2]]:
    # dy = 50. * (_cos - y)
    # ddy = 50. * (- _sin - dy: F64Array[int])
    return np.array(((0., -50.),))
# ======================================================================
def diffs2(x: float) -> tuple[Y1, Y1,Y1, Y1]:
    x = np.array((x,))
    y = np.exp(-15. * x)
    dy = -15. * y
    ddy = -15. * dy
    dddy = -15. * ddy
    return (y, dy, ddy, dddy)
# ======================================================================
def ddy2(x: float, y: Y1, dy: Y1) -> Y1:
    return -15. * dy
# ======================================================================
def jac2(x: float, y: Y1, dy: Y1
         ) -> F64Array[L[1], L[2]]:
    return np.array(((0., -15.),))
# ======================================================================
def diffs3(x: float) -> tuple[Y1, Y1,Y1, Y1]:
    x = np.array((x,))
    a = 20.
    exp = np.exp(-a * x)
    y = 1./(exp + 1.)
    dy = a * y * y * exp
    # = a * (2. * y * dy * exp - a * y * y * exp)
    # = a * y * exp * (2. * dy - a * y)
    # = a * y * exp * (2. * a * y * y * exp - a * y)
    # = a * dy * (2 * y * exp - 1)
    ddy = a * dy * (2. * y * exp - 1)
    dddy = a * (ddy * (2 * y * exp - 1) + dy * (2. * exp * (dy - a * y)))
    return (y, dy, ddy, dddy)
# ======================================================================
def ddy3(x: float, y: Y1, dy: Y1) -> Y1:
    x = np.array((x,))
    a = 20.
    exp = np.exp(-a * x)
    return a * dy[0] * (2. * y[0] * exp - 1)
# ======================================================================
def jac3(x: float, y: Y1, dy: Y1) -> F64Array[L[1], L[2]]:
    a = 20.
    exp = np.exp(-a * x)
    return a * np.array(((dy[0] * 2. * exp, (2. * y[0] * exp - 1)),))
# ======================================================================
def diffs4(x: float) -> tuple[Y1, Y1,Y1, Y1]:
    x = np.array((x,))
    exp_1 = np.exp(-x)
    exp_1000 = np.exp(-1e3 * x)
    y = 2. * exp_1 - exp_1000
    dy = -y + (1e3 - 1. ) * exp_1000
    # ddy = y + (1. - 1e6) * exp_1000
    # (dy + y) / (1e3 - 1. ) = exp_1000
    # ddy = y + (1. - 1e6) / (1e3 - 1. ) * (dy + y)
    a = (1. - 1e6) / (1e3 - 1. )
    ddy = (1. - a) * y + a * dy
    dddy = - y + (1e9 - 1. ) * exp_1000
    return (y, dy, ddy, dddy)
# ======================================================================
def ddy4(x: float, y: Y1, dy: Y1) -> Y1:
    a = (1. - 1e6) / (1e3 - 1. )
    return (1. - a) * y + a * dy
# ======================================================================
def jac4(x: float, y: Y1, dy: Y1
         ) -> F64Array[L[1], L[2]]:
    # y = 2. * exp_1 - exp_1000
    # ddy = y + (1. - 1e6) * exp_1000
    a = (1. - 1e6) / (1e3 - 1. )
    return np.array(((1. - a, a),))
