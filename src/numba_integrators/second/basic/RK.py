"""Basic RK integrators implemented with numba jitclass."""
import numba as nb
import numpy as np

from ..._aux import calc_eps
from ..._aux import calc_error_norm2
from ..._aux import IterableNamespace
from ..._aux import jitclass_from_dict
from ..._aux import MAX_FACTOR
from ..._aux import MIN_FACTOR
from ..._aux import nbA
from ..._aux import nbARO
from ..._aux import nbDec
from ..._aux import nbDecC
from ..._aux import npAFloat64
from ..._aux import RK23_params
from ..._aux import RK45_params
from ..._aux import SAFETY
from ..._aux import SolverBase
from ..._aux import step_prep
from ._second_basic_aux import nbODE2_type
from ._second_basic_aux import ODE2Type
from ._second_basic_aux import select_initial_step
from ._second_basic_aux import Solver2
# ======================================================================
@nbDecC
def substeps1(fun, x_old, y_old, h, A, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
    """Basic replicating the first order substeps."""
    dy_old = K[0, 0]
    Dx = C[0] * h
    dy = dy_old + Dx * K[1, 0]
    K[0, 1] = dy
    K[1, 1] = fun(x_old + Dx, y_old + Dx * dy_old, dy)

    for s in range(2, n_stages):
        ah = A[s - 2,:s] * h
        dy = dy_old + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x_old + C[s - 1] * h, y_old + np.dot(ah, K[0, :s]), dy)

    # Last step
    x = x_old + h
    ah = A[-2,:-1] * h
    y = y_old + np.dot(ah, K[0, :-1])
    dy = dy_old + np.dot(ah, K[1, :-1])
    K[0, -1] = dy
    K[1, -1] = fun(x, y, dy)
    return x, y
# # ----------------------------------------------------------------------
# @nbDecC
# def substeps2(fun, x_old, y_old,  h, A, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
#     _A = A.copy()
#     for i in range(1, len(C)):
#         _A[i] /= C[i]
#     for s in range(1, n_stages):
#         Dx = C[s] * h
#         dy = dy_old + np.dot(_A[s,:s] * Dx, K[1, :s])
#         K[s] = dy
#         # scaler = C[s] / C[s+1]
#         Dy1 = np.dot(_A[s,:s] * Dx, K[:s])
#         # k = K[:s+1]
#         # a = _A[s+1,s+1]
#         Dy2 = np.dot(_A[s+1, :s] * Dx, K[:s])
#         y = y_old + 0.5 * (Dy1 + Dy2)
#         K[1, s] = fun(x_old + Dx, y, dy)
#     # s = end
#     # Dx = C[s] * h
#     # dy = dy_old + np.dot(_A[s,:s] * Dx, K[1, :s])
#     # K[s] = dy
#     # # scaler = C[s] / C[s+1]
#     # K[1, s] = fun(x_old + Dx, y_old + np.dot(_A[s,:s] * Dx, K[:s]), dy)

#     Bh = B * h

#     Ddy = np.dot(Bh, K[1, :-1])
#     dy = dy_old + Ddy
#     y = y_old + np.dot(Bh, K[:-1])# + adjustment

#     K[-1] = dy
#     K[1, -1] = fun(x_old + h, y, dy)
#     return y
# ----------------------------------------------------------------------
@nbDecC
def _poly2_estimation(Dx: np.float64,
                      dy0: npAFloat64,
                      ddy0: npAFloat64) -> tuple[npAFloat64, npAFloat64]:
    return (Dx * ((0.5 * Dx) * ddy0 + dy0), ddy0 * Dx)
# ----------------------------------------------------------------------
@nbDecC
def substeps3(fun, x_old, y_old, h, A, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
    """First substep with second degree polynomial."""
    dy_old = K[0, 0]
    Dx = C[0] * h

    Dy, Ddy = _poly2_estimation(Dx, dy_old, K[1, 0])
    dy = dy_old + Ddy
    K[0, 1] = dy
    K[1, 1] = fun(x_old + Dx, y_old + Dy, dy)

    for s in range(2, n_stages):
        Dx = C[s-1] * h
        ah = A[s - 2,:s] * Dx
        dy = dy_old + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x_old + Dx, y_old + np.dot(ah, K[0, :s]), dy)

    # Last step
    x = x_old + h
    ah = A[-2,:-1] * h
    y = y_old + np.dot(ah, K[0, :-1])
    dy = dy_old + np.dot(ah, K[1, :-1])
    K[0, -1] = dy
    K[1, -1] = fun(x, y, dy)
    return x, y
# ----------------------------------------------------------------------
# @nbDecC
# def substeps4(fun, x_old, y_old, h, A, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
#     """First substep with second degree polynomial, second substep with fifth
#     degree polynomial."""

#     # First substep
#     dy_old = K[0, 0]
#     ddy_old = K[1, 0]
#     Dx = C[1] * h

#     dy = dy_old + ddy_old * Dx

#     p2 = 0.5 * ddy_old
#     p1 = dy_old
#     p0 = y_old

#     y = Dx * (p2 * Dx + p1) + p0
#     K[0, 1] = dy
#     K[1, 1] = fun(x_old + Dx, y, dy)

#     # Second substep
#     Dddy = (K[1, 1] - ddy_old)

#     _Dx = 1./Dx
#     _Dx_2 = _Dx * _Dx

#     p3 = Dddy * 0.5 * _Dx
#     p4 = - Dddy * _Dx_2
#     p5 = _Dx_2 * p3

#     Dx = C[2] * h

#     dy = Dx * (Dx * (Dx * (5. * p5 * Dx + 4. * p4) + 3. * p3) + 2. * p2) + p1
#     y = Dx * (Dx * (Dx * (Dx * (p5 * Dx + p4) + p3) + p2) + p1) + p0
#     K[0, 2] = dy
#     K[1, 2] = fun(x_old + Dx, y, dy)
#     # print('poly', y - np.sin(x_old + Dx), dy)
#     s = 2
#     # ah = A[s,:s] * h
#     # dy = dy_old + np.dot(ah, K[1, :s])
#     # K[s] = dy
#     # K[1, s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[:s]), dy)
#     # print( y_old + np.dot(ah, K[:s]) - np.sin(x_old + C[s] * h), dy)
#     for s in range(3, n_stages):
#         ah = A[s,:s] * h
#         dy = dy_old + np.dot(ah, K[1, :s])
#         y = y_old + np.dot(ah, K[0, :s])
#         K[0, s] = dy
#         K[1, s] = fun(x_old + C[s] * h, dy)
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
# def substeps5(fun, x_old, y_old, h, A, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
#     """First substep with fifth degree polynomial."""
#     dy_old = K[0, 0]
#     ddy_old = K[1, 0]
#     Dx = C[1] * h
#     if h_prev > 0.:
#         p0, p1, p2, p3, p4, p5 = make_poly5(y_old, dy_old, ddy_old,
#                                             -h_prev, y_prev,  dy_prev, ddy_prev)

#         dy = Dx * (Dx * (Dx * ((5. * Dx) * p5 + 4. * p4) + 3. * p3) + 2. * p2) + p1
#         y = Dx * (Dx * (Dx * (Dx * (Dx * p5 + p4) + p3) + p2) + p1) + p0
#         # p0, p1, p2, p3, p4, p5 = make_poly4_exp(y_old, dy_old, ddy_old,
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

#         Dy, Ddy = _poly2_estimation(Dx, dy_old, K[1, 0])
#         dy = dy_old + Ddy
#         y = y_old + Dy

#     K[0, 1] = dy
#     K[1, 1] = fun(x_old + Dx, y, dy)

#     for s in range(2, n_stages):
#         ah = A[s,:s] * h
#         dy = dy_old + np.dot(ah, K[1, :s])
#         y =  y_old + np.dot(ah, K[0, :s])
#         K[0, s] = dy
#         K[1, s] = fun(x_old + C[s] * h, y, dy)
#     return y
# ----------------------------------------------------------------------
# @nbDecC
# def substeps6(fun, x_old, y_old, h, A, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
#     """"""
#     dy_old = K[0, 0]
#     Dx = C[1] * h

#     Dy, Ddy = _poly2_estimation(Dx, dy_old, K[1, 0])

#     dy = dy_old + Ddy
#     K[0, 1] = dy
#     K[1, 1] = fun(x_old + Dx, y_old + Dy, dy)

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
#     y = y_old + Dy
#     dy = dy_old + Ddy
#     K[0, 2] = dy
#     K[1, 2] = fun(x_old + Dx, y_old + Dy, dy)

#     for s in range(3, n_stages):
#         ah = A[s,:s] * h
#         Dx = C[s] * h
#         Ddy = np.dot(ah, K[1, :s])

#         dy = dy_old + Ddy
#         y = y_old + np.dot(ah, K[0, :s])# + 0.5 * Ddy * Dx
#         K[0, s] = dy
#         K[1, s] = fun(x_old + Dx, y, dy)
#     return y
# ----------------------------------------------------------------------
@nbDecC
def substeps7(fun, x_old, y_old, h, A, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
    """First substep with second degree polynomial."""
    dy_old = K[0, 0]
    Dx = C[0] * h

    Dy, Ddy = _poly2_estimation(Dx, dy_old, K[1, 0])
    dy = dy_old + Ddy
    K[0, 1] = dy
    K[1, 1] = fun(x_old + Dx, y_old + Dy, dy)

    for s in range(2, n_stages):
        Dx = C[s-1] * h
        ah = A[s - 2,:s] * Dx
        dy = dy_old + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x_old + Dx, y_old + np.dot(ah, K[0, :s]), dy)

    # Last step
    x = x_old + h
    ah = A[-2,:-1] * h
    y = y_old + np.dot(ah, K[0, :-1])
    dy = dy_old + np.dot(ah, K[1, :-1])
    K[0, -1] = dy
    K[1, -1] = fun(x, y, dy)
    return x, y
# ----------------------------------------------------------------------
@nbDecC
def RKstep(fun: ODE2Type,
          direction: np.float64,
          x_old: np.float64,
          y_old: npAFloat64,
          y_prev: npAFloat64,
          h_abs: np.float64,
          h_prev: np.float64,
          min_step: np.float64,
          K: npAFloat64,
          n_stages: np.uint64,
          rtol: npAFloat64,
          atol: npAFloat64,
          A: npAFloat64,
          C: npAFloat64,
          error_exponent: np.float64) -> tuple[bool,
                                          np.float64,
                                          npAFloat64,
                                          np.float64,
                                          np.float64,
                                          np.uint64]:
    nfev = np.uint64(0)
    # Saving previous step info
    dy_prev = K[0, 0]
    ddy_prev = K[1, 0]
    # Prepping for next step
    K[0, 0] = K[0, -1]
    K[1, 0] = K[1, -1]
    y_old_abs = np.abs(y_old)
    dy_old_abs = np.abs(K[0, 0])
    _len = 1. / len(y_old)
    while True: # := not working
        h = direction * h_abs
        # _RK[1] core loop
        # len(K) == n_stages + 1
        x, y = substeps3(fun, x_old, y_old, h, A, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev)
        nfev += n_stages
        # print(x, y, K[0, -1])
        # Error calculation
        error_norm2 = (calc_error_norm2(
            np.dot(A[-1], K[0]), np.abs(y), y_old_abs, rtol, atol)
                       + calc_error_norm2(
            np.dot(A[-1], K[1]), np.abs(K[0, -1]), dy_old_abs, rtol, atol)
            ) * 0.5 * _len * h * h

        if error_norm2 < 1.:
            h_abs *= (MAX_FACTOR if error_norm2 == 0. else
                      min(MAX_FACTOR, SAFETY * error_norm2 ** error_exponent))
            if h_abs < min_step: # Due to the SAFETY, the step can shrink
                h_abs = min_step
            return True, y, x, h_abs, h, nfev # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm2 ** error_exponent)
            if h_abs < min_step:
                return False, y, x, h_abs, h, nfev # Too small step size
# ======================================================================
@jitclass_from_dict({'fun': nbODE2_type,
                     'K': nbA(3),
                     'y_prev': nbA(1)})
class _RK2(SolverBase):
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
                 error_exponent: np.float64,
                 n_stages: np.uint64,
                 A: npAFloat64,
                 C: npAFloat64):
        self.n_stages = n_stages
        self.A = A
        self.C = C
        self.fun = fun
        self.x = x0
        self.y = y0
        self.y_prev = y0
        self.x_bound = x_bound
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.error_exponent = error_exponent

        _1 = np.uint64(1)

        self.K = np.zeros((2, self.n_stages + _1, len(y0)), dtype = np.float64)
        self.K[0, -1] = dy0
        self.K[1, -1] = self.fun(self.x, self.y, self.dy)
        self._nfev = _1
        self.direction = 1. if x_bound == x0 else np.sign(x_bound - x0)

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.x, y0, dy0, self.K[1, -1], self.direction,
                self.error_exponent, self.rtol, self.atol)
            self._nfev += _1
        else:
            self.h_abs = np.abs(first_step)

        min_step = 8. * calc_eps(self.x, self.direction)
        if self.h_abs < min_step:
            self.h_abs = min_step

        self.step_size = 0.
    # ------------------------------------------------------------------
    def step(self) -> bool:
        h_end = self.x_bound - self.x
        if self.direction * h_end > 0.: # x_bound has not been reached
            min_step, h_abs = step_prep(self.x,
                                        self.h_abs,
                                        self.direction,
                                        h_end,
                                        self.max_step)
            y_prev = self.y
            (valid,
            self.y,
            self.x,
            self.h_abs,
            self.step_size,
            nfev) = RKstep(self.fun,
                            self.direction,
                            self.x,
                            self.y,
                            self.y_prev,
                            h_abs,
                            self.step_size,
                            min_step,
                            self.K,
                            self.n_stages,
                            self.rtol,
                            self.atol,
                            self.A,
                            self.C,
                            self.error_exponent)
            self.y_prev = y_prev
            self._nfev += nfev
            return valid
        return False
    # ------------------------------------------------------------------
    @property
    def dy(self) -> npAFloat64:
        return self.K[0, -1]
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[np.float64, npAFloat64, npAFloat64]:
        return self.x, self.y, self.K[0, -1]
# ======================================================================
@nbDec(cache = False) # Some issue in making caching jitclasses
def init_RK2(fun: ODE2Type,
            x0: np.float64,
            y0: npAFloat64,
            dy0: npAFloat64,
            x_bound: np.float64,
            max_step: np.float64,
            rtol: npAFloat64,
            atol: npAFloat64,
            first_step: np.float64,
            solver_params) -> _RK2:
    return _RK2(fun, x0, y0, dy0, x_bound, max_step, rtol, atol, first_step,
               *solver_params)
# ======================================================================
class _Solver2_RK(Solver2):
    _init = init_RK2
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
