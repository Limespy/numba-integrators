"""Basic RK integrators implemented with numba jitclass."""
import numba as nb
import numpy as np

from ..._aux import calc_error
from ..._aux import IterableNamespace
from ..._aux import jitclass_from_dict
from ..._aux import MIN_FACTOR
from ..._aux import nbA
from ..._aux import nbARO
from ..._aux import nbDec
from ..._aux import nbDecC
from ..._aux import npAFloat64
from ..._aux import RK23_params
from ..._aux import RK45_params
from ..._aux import SAFETY
from ._second_basic_aux import nbODE2_type
from ._second_basic_aux import ODE2Type
from ._second_basic_aux import SecondBasicSolverBase
from ._second_basic_aux import select_initial_step
from ._second_basic_aux import Solver2
# ======================================================================
@nbDecC
def _step1(fun, x0, y0, h, K, n_stages, A, C):
    """Basic replicating the first order substeps."""
    dy0 = K[0, 0]
    Dx = C[0] * h
    dy = dy0 + Dx * K[1, 0]
    K[0, 1] = dy
    K[1, 1] = fun(x0 + Dx, y0 + Dx * dy0, dy)

    for s in range(2, n_stages):
        ah = A[s - 2,:s] * h
        dy = dy0 + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x0 + C[s - 1] * h, y0 + np.dot(ah, K[0, :s]), dy)

    # Last step
    x = x0 + h
    ah = A[-2,:-1] * h
    y = y0 + np.dot(ah, K[0, :-1])
    dy = dy0 + np.dot(ah, K[1, :-1])
    K[0, -1] = dy
    K[1, -1] = fun(x, y, dy)
    return x, y
# # ----------------------------------------------------------------------
# @nbDecC
# def substeps2(fun, x0, y0,  h, K, n_stages, A, C):
#     _A = A.copy()
#     for i in range(1, len(C)):
#         _A[i] /= C[i]
#     for s in range(1, n_stages):
#         Dx = C[s] * h
#         dy = dy0 + np.dot(_A[s,:s] * Dx, K[1, :s])
#         K[s] = dy
#         # scaler = C[s] / C[s+1]
#         Dy1 = np.dot(_A[s,:s] * Dx, K[:s])
#         # k = K[:s+1]
#         # a = _A[s+1,s+1]
#         Dy2 = np.dot(_A[s+1, :s] * Dx, K[:s])
#         y = y0 + 0.5 * (Dy1 + Dy2)
#         K[1, s] = fun(x0 + Dx, y, dy)
#     # s = end
#     # Dx = C[s] * h
#     # dy = dy0 + np.dot(_A[s,:s] * Dx, K[1, :s])
#     # K[s] = dy
#     # # scaler = C[s] / C[s+1]
#     # K[1, s] = fun(x0 + Dx, y0 + np.dot(_A[s,:s] * Dx, K[:s]), dy)

#     Bh = B * h

#     Ddy = np.dot(Bh, K[1, :-1])
#     dy = dy0 + Ddy
#     y = y0 + np.dot(Bh, K[:-1])# + adjustment

#     K[-1] = dy
#     K[1, -1] = fun(x0 + h, y, dy)
#     return y
# ----------------------------------------------------------------------
@nbDecC
def _poly2_estimation(Dx: np.float64,
                      dy0: npAFloat64,
                      ddy0: npAFloat64) -> tuple[npAFloat64, npAFloat64]:
    return (Dx * ((0.5 * Dx) * ddy0 + dy0), ddy0 * Dx)
# ----------------------------------------------------------------------
@nbDecC
def _step3(fun, x0, y0, h, K, n_stages, A, C):
    """First substep with second degree polynomial."""
    dy0 = K[0, 0]
    Dx = C[0] * h

    Dy, Ddy = _poly2_estimation(Dx, dy0, K[1, 0])
    dy = dy0 + Ddy
    K[0, 1] = dy
    K[1, 1] = fun(x0 + Dx, y0 + Dy, dy)

    for s in range(2, n_stages):
        Dx = C[s-1] * h
        ah = A[s - 2,:s] * Dx
        dy = dy0 + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x0 + Dx, y0 + np.dot(ah, K[0, :s]), dy)

    # Last step
    x = x0 + h
    ah = A[-2,:-1] * h
    y = y0 + np.dot(ah, K[0, :-1])
    dy = dy0 + np.dot(ah, K[1, :-1])
    K[0, -1] = dy
    K[1, -1] = fun(x, y, dy)
    return x, y
# ----------------------------------------------------------------------
# @nbDecC
# def _step4(fun, x0, y0, h, K, n_stages, A, C):
#     """First substep with second degree polynomial, second substep with fifth
#     degree polynomial."""

#     # First substep
#     dy0 = K[0, 0]
#     ddy0 = K[1, 0]
#     Dx = C[1] * h

#     dy = dy0 + ddy0 * Dx

#     p2 = 0.5 * ddy0
#     p1 = dy0
#     p0 = y0

#     y = Dx * (p2 * Dx + p1) + p0
#     K[0, 1] = dy
#     K[1, 1] = fun(x0 + Dx, y, dy)

#     # Second substep
#     Dddy = (K[1, 1] - ddy0)

#     _Dx = 1./Dx
#     _Dx_2 = _Dx * _Dx

#     p3 = Dddy * 0.5 * _Dx
#     p4 = - Dddy * _Dx_2
#     p5 = _Dx_2 * p3

#     Dx = C[2] * h

#     dy = Dx * (Dx * (Dx * (5. * p5 * Dx + 4. * p4) + 3. * p3) + 2. * p2) + p1
#     y = Dx * (Dx * (Dx * (Dx * (p5 * Dx + p4) + p3) + p2) + p1) + p0
#     K[0, 2] = dy
#     K[1, 2] = fun(x0 + Dx, y, dy)
#     # print('poly', y - np.sin(x0 + Dx), dy)
#     s = 2
#     # ah = A[s,:s] * h
#     # dy = dy0 + np.dot(ah, K[1, :s])
#     # K[s] = dy
#     # K[1, s] = fun(x0 + C[s] * h, y0 + np.dot(ah, K[:s]), dy)
#     # print( y0 + np.dot(ah, K[:s]) - np.sin(x0 + C[s] * h), dy)
#     for s in range(3, n_stages):
#         ah = A[s,:s] * h
#         dy = dy0 + np.dot(ah, K[1, :s])
#         y = y0 + np.dot(ah, K[0, :s])
#         K[0, s] = dy
#         K[1, s] = fun(x0 + C[s] * h, dy)
#     return y
# # ----------------------------------------------------------------------
# @nbDecC
# def make_poly5(y0, dy0, ddy0, Dx, y, dy, ddy):
#     p2 = 0.5 * ddy0
#     p1 = dy0
#     p0 = y0
#     # Dddy = ddy_1 - ddy0

#     # _Dx = -1./Dx
#     # _Dx_2 = _Dx * _Dx

#     Dx2 = Dx * Dx
#     Dx3 = Dx * Dx2
#     Dx4 = Dx2 * Dx2
#     Dx5 = Dx3 * Dx2
#     A = np.array(((Dx3, Dx4, Dx5),
#                     (3. * Dx2, 4. * Dx3, 5. * Dx4),
#                     (6. * Dx, 12. * Dx2, 20. * Dx3)))

#     b = np.zeros((3, len(y)))
#     b[0] = y - (p0 + p1 * Dx + p2 * Dx2)
#     b[1] = dy - (p1 + ddy0 * Dx)
#     b[2] = ddy - ddy0
#     # b = np.array((y - (p0 + p1 * Dx + p2 * Dx2),
#     #               dy - (p1 + ddy0 * Dx),
#     #               ddy - ddy0))

#     p3, p4, p5 = np.linalg.solve(A, b)
#     return p0, p1, p2, p3, p4, p5
# # ----------------------------------------------------------------------
# @nbDecC
# def make_poly4_exp(y0, dy0, ddy0, Dx, y, dy, ddy):
#     Dx2 = Dx * Dx
#     Dx3 = Dx * Dx2
#     Dx4 = Dx2 * Dx2
#     expm1x = np.expm1(Dx)
#     A = np.array(((Dx3, Dx4, expm1x - 0.5 * Dx2 - Dx),
#                     (3. * Dx2, 4. * Dx3, expm1x - Dx),
#                     (6. * Dx, 12. * Dx4, expm1x)))

#     b = np.zeros((3, len(y)))

#     b[0] = y - (y0 + dy0 * Dx + 0.5 * ddy0 * Dx2)
#     b[1] = dy - (ddy0 * Dx + dy0)
#     b[2] = ddy - ddy0

#     p3, p4, p5 = np.linalg.solve(A, b)

#     p0 = y0 - p5
#     p1 = dy0 - p5
#     p2 = 0.5 * (ddy0 - p5)
#     return p0, p1, p2, p3, p4, p5
# # ----------------------------------------------------------------------
# @nbDecC
# def _step5(fun, x0, y0, h, K, n_stages, A, C):
#     """First substep with fifth degree polynomial."""
#     dy0 = K[0, 0]
#     ddy0 = K[1, 0]
#     Dx = C[1] * h
#     if h_prev > 0.:
#         p0, p1, p2, p3, p4, p5 = make_poly5(y0, dy0, ddy0,
#                                             -h_prev, y_prev,  dy_prev, ddy_prev)

#         dy = Dx * (Dx * (Dx * ((5. * Dx) * p5 + 4. * p4) + 3. * p3) + 2. * p2) + p1
#         y = Dx * (Dx * (Dx * (Dx * (Dx * p5 + p4) + p3) + p2) + p1) + p0
#         # p0, p1, p2, p3, p4, p5 = make_poly4_exp(y0, dy0, ddy0,
#         #                                     -h_prev, y_prev,  dy_prev, ddy_prev)
#         # Dx2 = Dx * Dx
#         # Dx3 = Dx2 * Dx
#         # Dx4 = Dx2 * Dx2
#         # expx = np.exp(Dx)
#         # dy = (p5 * expx
#         #     + p4 * (4. * Dx4)
#         #     + p3 * (3. * Dx2)
#         #     + p2 * (2. * Dx)
#         #     + p1)
#         # y = p5 * expx + p4 * Dx4 + p3 * Dx3 + p2 * Dx2 + p1 * Dx + p0
#     else:

#         Dy, Ddy = _poly2_estimation(Dx, dy0, K[1, 0])
#         dy = dy0 + Ddy
#         y = y0 + Dy

#     K[0, 1] = dy
#     K[1, 1] = fun(x0 + Dx, y, dy)

#     for s in range(2, n_stages):
#         ah = A[s,:s] * h
#         dy = dy0 + np.dot(ah, K[1, :s])
#         y =  y0 + np.dot(ah, K[0, :s])
#         K[0, s] = dy
#         K[1, s] = fun(x0 + C[s] * h, y, dy)
#     return y
# ----------------------------------------------------------------------
# @nbDecC
# def _step6(fun, x0, y0, h, K, n_stages, A, C):
#     """"""
#     dy0 = K[0, 0]
#     Dx = C[1] * h

#     Dy, Ddy = _poly2_estimation(Dx, dy0, K[1, 0])

#     dy = dy0 + Ddy
#     K[0, 1] = dy
#     K[1, 1] = fun(x0 + Dx, y0 + Dy, dy)

#     Dx = C[2] * h

#     Dy1, Ddy1 = _poly2_estimation(Dx, K[0, 0], K[1, 0])

#     Dy2, Ddy2 = _poly2_estimation(Dx, K[0, 1], K[1, 1])
#     a1, a2 = A[2,:2] / C[2]
#     Ah = A[2,:2] * h

#     Ddy = np.dot(Ah, K[1, :2])
#     Dy = np.dot(Ah, K[0, :2])
#     Ddy = a1 * Ddy1 + a2 * Ddy2
#     Dy = a1 * Dy1 + a2 * Dy2
#     # print(Dy, a1 * Dy1 + a2 * Dy2)
#     # print(Ddy, a1 * Ddy1 + a2 * Ddy2)
#     y = y0 + Dy
#     dy = dy0 + Ddy
#     K[0, 2] = dy
#     K[1, 2] = fun(x0 + Dx, y0 + Dy, dy)

#     for s in range(3, n_stages):
#         ah = A[s,:s] * h
#         Dx = C[s] * h
#         Ddy = np.dot(ah, K[1, :s])

#         dy = dy0 + Ddy
#         y = y0 + np.dot(ah, K[0, :s])# + 0.5 * Ddy * Dx
#         K[0, s] = dy
#         K[1, s] = fun(x0 + Dx, y, dy)
#     return y
# ----------------------------------------------------------------------
@nbDecC
def _RK_adaptive_step(fun: ODE2Type,
          x0: np.float64,
          y0: npAFloat64,
          h: np.float64,
          h_abs_min: np.float64,
          K: npAFloat64,
          rtol: npAFloat64,
          atol: npAFloat64,
          _len: np.float64,
          n_stages: np.int64,
          A: npAFloat64,
          C: npAFloat64,
          error_exponent: np.float64
          ) -> tuple[np.float64, np.float64, npAFloat64, np.float64, np.int64]:
    # Prepping for next step
    nfev = np.int64(n_stages)
    y0_abs = np.abs(y0)
    dy0_abs = np.abs(K[0, 0])
    _len_2 = 0.5 * _len

    x, y = _step3(fun, x0, y0, h, K, n_stages, A, C)
    error = (calc_error(np.dot(A[-1], K[0]), y, y0_abs, rtol, atol)
             + calc_error(np.dot(A[-1], K[1]), K[0, -1], dy0_abs, rtol, atol)
            ) * _len_2 * h * h
    while error > 1. and abs(h) > h_abs_min: # := not working
        h *= max(MIN_FACTOR, SAFETY * error ** error_exponent)

        # RK core loop
        x, y = _step3(fun, x0, y0, h, K, n_stages, A, C)
        error = (calc_error(np.dot(A[-1], K[0]), y, y0_abs, rtol, atol)
                 + calc_error(np.dot(A[-1], K[1]), K[0, -1], dy0_abs, rtol, atol)
            ) * _len_2 * h * h
        nfev += n_stages
    return error, x, y, h, nfev
# ======================================================================
@jitclass_from_dict({'fun': nbODE2_type,
                     'K': nbA(3),
                     'dy': nbA(1)})
class _RK_2(SecondBasicSolverBase):
    """Base class for explicit Runge-Kutta methods."""
    def __init__(self,
                 fun: ODE2Type,
                 x0: np.float64,
                 y0: npAFloat64,
                 dy0: npAFloat64,
                 x_bound: np.float64,
                 max_step: np.float64,
                 rtol: npAFloat64,
                 atol: npAFloat64,
                 first_step: np.float64,
                 n_stages: np.int64,
                 A: npAFloat64,
                 C: npAFloat64,
                 error_exponent: np.float64):
        self.fun = fun
        self.x = x0
        self.y = y0
        self.dy = dy0
        self.x_bound = x_bound
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.n_stages = n_stages
        self.A = A
        self.C = C
        self.error_exponent = error_exponent

        self.reboot(first_step)
    # ------------------------------------------------------------------
    def _init_K(self, y_len: int):
        self.K = np.zeros((2, self.n_stages + np.int64(1), y_len),
                          dtype = np.float64)
        self.K[0, 0] = self.dy
        self.K[1, 0] = self.fun(self.x, self.y, self.dy)
    # ------------------------------------------------------------------
    def _select_initial_step(self, direction: np.float64) -> np.float64:
        return select_initial_step(self.fun, self.x, self.y, self.dy,
                                   self.K[1, 0], direction,
                                   self.error_exponent, self.rtol, self.atol)
    # ------------------------------------------------------------------
    def _step(self) -> tuple[np.float64, np.int64]:
        (error,
         self.x,
         self.y,
         self.step_size,
         nfev) = _RK_adaptive_step(self.fun,
                        self.x,
                        self.y,
                        self._h_next,
                        self._h_abs_min,
                        self.K,
                        self.rtol,
                        self.atol,
                        self._len,
                        self.n_stages,
                        self.A,
                        self.C,
                        self.error_exponent)
        self.K[0, 0] = self.K[0, -1]
        self.K[1, 0] = self.K[1, -1]
        self.dy = self.K[0, 0]
        return error, nfev
    # ------------------------------------------------------------------
    @property
    def ddy(self) -> npAFloat64:
        return self.K[1, -1]
# ======================================================================
@nbDec(cache = False) # Some issue in making caching jitclasses
def init_RK_2(fun: ODE2Type,
            x0: np.float64,
            y0: npAFloat64,
            dy0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params) -> _RK_2:
    return _RK_2(fun, x0, y0, dy0, x_bound, max_step, rtol, atol, first_step,
               *solver_params)
# ======================================================================
class _Solver2_RK(Solver2):
    _init = init_RK_2
# ----------------------------------------------------------------------
class RK23_2(_Solver2_RK):
    _solver_params = RK23_params
# ----------------------------------------------------------------------
class RK45_2(_Solver2_RK):
    _solver_params = RK45_params
# ----------------------------------------------------------------------
class Solvers(IterableNamespace):
    RK23_2 = RK23_2
    RK45_2 = RK45_2
