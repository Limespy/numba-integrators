import numpy as np

from .._aux import nbDecFC
from .._aux import norm
from .._aux import npAFloat64
# ======================================================================
@nbDecFC
def calc_h0(y0: npAFloat64,
            dy0: npAFloat64,
            direction: np.float64,
            scale: npAFloat64):
    d0 = norm(y0 / scale)
    d1 = norm(dy0 / scale)

    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1

    y1 = y0 + h0 * direction * dy0
    return h0, y1, d1
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
                else (max(d1, d2) * 100.) **(2.* error_exponent)))
# ----------------------------------------------------------------------
