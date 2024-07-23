from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

from .._aux import nbDecFC
from .._aux import norm
from .._aux import npAFloat64
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
class RKN_Params(NamedTuple):
    n_stages: np.int64
    alpha: npAFloat64
    beta: npAFloat64
    gamma: npAFloat64
    error_exponent: np.float64
# ----------------------------------------------------------------------
def make_RKN_params(order: int | float,
                   beta: Iterable[Iterable[float]],
                   gamma: Iterable[float],
                   alpha: Iterable[float] = ()) -> RKN_Params:
    if not alpha:
        alpha = [sum(row) for row in beta]
    _a, _b, _g = (np.array(M, np.float64) for M in (alpha, beta, gamma))
    _a_inv = 1. / _a.reshape(-1,1)[1:]

    # Condition from the publication
    # \Sigma_{\lambda=1}^{\kappa-1} \beta[\kappa,\lambda] \cdot \alpha[\lambda]
    # = alpha[\kappa]^2 / 2
    # for \kappa = [2, 8]
    for k in range(2, 7):
        ref_a = 0.5 * _a[k-1]**2
        ref_b = np.dot(_b[k-2, 1:k], _a[:k-1])
        if abs(ref_a -ref_b) > 1e-14 * ref_a:
          raise ValueError(k, ref_b, ref_a)

    # Condition from the publication
    # \Sigma_{\lambda=1}^{\kappa-1} \beta[\kappa,\lambda] \cdot \alpha[\lambda]^2
    # = alpha[\kappa]^4 / 3
    # for \kappa = [2, 8]
    for k in range(2, 7):
        ref_a = _a[k-1]**3 / 3.
        ref_b = np.dot(_b[k-2, 1:k], _a[:k-1] * _a[:k-1])
        if abs(ref_a -ref_b) > 1e-14 * ref_a:
          raise ValueError(k, ref_b, ref_a)
    # Normalising sum to 1
    _b[:-2] *= _a_inv

    # sum to 0.5
    _g[:-2] *= (_a_inv * _a_inv)
    # _g[-2:] *= 0.5 # this already is 0.5 ???
    b_sum = np.sum(_b, axis = 1)
    g_sum = np.sum(_g, axis = 1)
    if not (np.allclose(b_sum, 1., 1e-16, 1e-16)
            and np.allclose(g_sum, 0.5, 1e-16, 1e-16)):
        print(b_sum)
        print(g_sum)
        raise ValueError

    return RKN_Params(np.int64(len(beta) + 1),
                     _a, _b, _g, np.float64(-0.5 / (order + 1.)),)
# ----------------------------------------------------------------------
RKN56_params = make_RKN_params(5.,
((1/10,         3/10,         0,            0,
  0,            0,            0,            0),
 (3/20,         0,            9/20,         0,
  0,            0,            0,            0),
 (9/40,         0,            0,            27/40,
  0,            0,            0,            0),
 (11/48,        0,            0,            5/8,
  -5/48,        0,            0,            0),
 (27112/194481, 0,            0,            56450/64827,
  80000/194481, -24544/21609, 0,            0),
 (-26033/41796, 0,            0,            -236575/38313,
  -14500/10449, 275936/45279, 228095/73788, 0),
 (7/81,         0,            0,            0,
  -250/3483,    160/351,      2401/5590,    1/10)),
((1/25,            1/25,           0,               0,
  0,               0,              0,               0),
 (9/160,           81/800,         9/400,           0,
  0,               0,              0,               0),
 (81/640,          0,              729/3200,        81/1600,
  0,               0,              0,               0),
 (11283/88064,     0,              3159/88064,      7275/44032,
  -33/688,         0,              0,               0),
 (6250/194481,     0,              0,               0,
  -3400/194481,    1696/64827,     0,               0),
 (-6706/45279,     0,              0,               0,
  1047925/1946997, -147544/196209, 1615873/1874886, 0),
 (31/360,          0,              0,               0,
  0,               64/585,         2401/7800,       -1/300)),
(4/15, 2/5, 3/5, 9/10, 3/4, 2/7))
