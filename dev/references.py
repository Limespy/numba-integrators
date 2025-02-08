import numpy as np
# ======================================================================
def diffs0(x):
    y = np.sin(x)
    dy = np.cos(x)
    return np.array((y, dy, -y, -dy))
# ======================================================================
def ddy0(x, y, dy):
    return -y
# ======================================================================
def jac0(x, y, dy):
    return np.array(((-1., 0.),))
# ======================================================================
def diffs1(x):
    _sin = np.sin(x)
    _cos = np.cos(x)
    y = 50. * (_sin + 50. * _cos - 50. * np.exp(-50. * x)) / 2501.
    dy = 50. * (_cos - y)
    ddy = 50. * (- _sin - dy)
    dddy = 50. * (- _cos - ddy)
    return np.array((y, dy, ddy, dddy))
# ======================================================================
def ddy1(x, y, dy):
    return 50. * (- np.sin(x) - dy)
# ======================================================================
def jac1(x, y, dy):
    # dy = 50. * (_cos - y)
    # ddy = 50. * (- _sin - dy)
    return np.array(((0., -50.),))
# ======================================================================
def diffs2(x):
    y = np.exp(-15. * x)
    dy = -15. * y
    ddy = -15. * dy
    dddy = -15. * ddy
    return np.array((y, dy, ddy, dddy))
# ======================================================================
def jac2(x, y, dy):
    return np.array(((0., -15.),))
# ======================================================================
def diffs3(x):
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
    return np.array((y, dy, ddy, dddy))
# ======================================================================
def jac3(x, y, dy):
    a = 20.
    exp = np.exp(-a * x)
    return a * np.array(((dy * 2. * exp, (2. * y * exp - 1)),))
# ======================================================================
def diffs4(x):
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
    return np.array((y, dy, ddy, dddy))
# ======================================================================
def jac4(x, y, dy):
    # y = 2. * exp_1 - exp_1000
    # ddy = y + (1. - 1e6) * exp_1000
    a = (1. - 1e6) / (1e3 - 1. )
    return np.array(((1. - a, a),))
