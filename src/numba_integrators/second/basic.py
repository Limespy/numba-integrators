"""Basic RK integrators implemented with numba jitclass."""
import numba as nb
import numpy as np

from .._aux import Arrayable
from .._aux import calc_error_norm2
from .._aux import calc_tolerance
from .._aux import convert
from .._aux import h_prep
from .._aux import IS_CACHE
from .._aux import IterableNamespace
from .._aux import jitclass_from_dict
from .._aux import MAX_FACTOR
from .._aux import MIN_FACTOR
from .._aux import nbA
from .._aux import nbARO
from .._aux import nbDecC
from .._aux import nbDecFC
from .._aux import nbODE2_type
from .._aux import norm
from .._aux import npAFloat64
from .._aux import ODE2Type
from .._aux import RK23_params
from .._aux import RK45_params
from .._aux import RK_Params
from .._aux import SAFETY
from .._aux import SolverBase
from .._aux import step_prep
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
@nb.njit(nb.float64(nbODE2_type,
                    nb.float64,
                    nbA(1),
                    nbA(1),
                    nbA(1),
                    nb.float64,
                    nb.float64,
                    nbARO(1),
                    nbARO(1)),
         fastmath = True, cache = IS_CACHE)
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
# ----------------------------------------------------------------------
@nbDecC
def substeps1(fun, x_old, y_old, h, A, B, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
    """Basic replivcating the first order substeps."""
    dy_old = K[0, 0]
    for s in range(1, n_stages):
        ah = A[s,:s] * h
        dy = dy_old + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[0, :s]), dy)
    Bh = B * h

    Ddy = np.dot(Bh, K[1, :-1])
    dy = dy_old + Ddy
    y = y_old + np.dot(Bh, K[0, :-1])# + adjustment

    K[0, -1] = dy
    K[1, -1] = fun(x_old + h, y, dy)
    return y
# # ----------------------------------------------------------------------
# @nbDecC
# def substeps2(fun, x_old, y_old,  h, A, B, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
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
def substeps3(fun, x_old, y_old, h, A, B, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
    """First substep with second degree polynomial."""
    dy_old = K[0, 0]
    ddy_old = K[1, 0]
    Dx = C[1] * h

    a = 0.5 * ddy_old
    b = dy_old
    c = y_old

    dy = dy_old + ddy_old * Dx
    y = Dx * (a * Dx + b) + c

    K[0, 1] = dy
    K[1, 1] = fun(x_old + Dx, y, dy)

    for s in range(2, n_stages):
        ah = A[s,:s] * h
        dy = dy_old + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[0, :s]), dy)

    Bh = B * h

    Ddy = np.dot(Bh, K[1, :-1])
    dy = dy_old + Ddy
    y = y_old + np.dot(Bh, K[0, :-1])

    K[0, -1] = dy
    K[1, -1] = fun(x_old + h, y, dy)
    return y
# ----------------------------------------------------------------------
@nbDecC
def substeps4(fun, x_old, y_old, h, A, B, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):

    # First substep
    dy_old = K[0, 0]
    ddy_old = K[1, 0]
    Dx = C[1] * h

    dy = dy_old + ddy_old * Dx

    p2 = 0.5 * ddy_old
    p1 = dy_old
    p0 = y_old

    y = Dx * (p2 * Dx + p1) + p0
    K[0, 1] = dy
    K[1, 1] = fun(x_old + Dx, y, dy)

    # Second substep
    Dddy = (K[1, 1] - ddy_old)

    _Dx = 1./Dx
    _Dx_2 = _Dx * _Dx

    p3 = Dddy * 0.5 * _Dx
    p4 = - Dddy * _Dx_2
    p5 = _Dx_2 * p3

    Dx = C[2] * h

    dy = Dx * (Dx * (Dx * (5. * p5 * Dx + 4. * p4) + 3. * p3) + 2. * p2) + p1
    y = Dx * (Dx * (Dx * (Dx * (p5 * Dx + p4) + p3) + p2) + p1) + p0
    K[0, 2] = dy
    K[1, 2] = fun(x_old + Dx, y, dy)
    # print('poly', y - np.sin(x_old + Dx), dy)
    s = 2
    # ah = A[s,:s] * h
    # dy = dy_old + np.dot(ah, K[1, :s])
    # K[s] = dy
    # K[1, s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[:s]), dy)
    # print( y_old + np.dot(ah, K[:s]) - np.sin(x_old + C[s] * h), dy)
    for s in range(3, n_stages):
        ah = A[s,:s] * h
        dy = dy_old + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[0, :s]), dy)
    Bh = B * h

    Ddy = np.dot(Bh, K[1, :-1])
    dy = dy_old + Ddy
    y = y_old + np.dot(Bh, K[0, :-1])# + adjustment

    K[0, -1] = dy
    K[1, -1] = fun(x_old + h, y, dy)
    return y
# ----------------------------------------------------------------------
@nbDecC
def make_poly5(y0, dy0, ddy0, Dx, y, dy, ddy):
    p2 = 0.5 * ddy0
    p1 = dy0
    p0 = y0
    # Dddy = ddy_1 - ddy0

    # _Dx = -1./Dx
    # _Dx_2 = _Dx * _Dx

    Dx2 = Dx * Dx
    Dx3 = Dx * Dx2
    Dx4 = Dx2 * Dx2
    Dx5 = Dx3 * Dx2
    A = np.array(((Dx3, Dx4, Dx5),
                    (3. * Dx2, 4. * Dx3, 5. * Dx4),
                    (6. * Dx, 12. * Dx2, 20. * Dx3)))

    b = np.zeros((3, len(y)))
    b[0] = y - (p0 + p1 * Dx + p2 * Dx2)
    b[1] = dy - (p1 + ddy0 * Dx)
    b[2] = ddy - ddy0
    # b = np.array((y - (p0 + p1 * Dx + p2 * Dx2),
    #               dy - (p1 + ddy0 * Dx),
    #               ddy - ddy0))

    p3, p4, p5 = np.linalg.solve(A, b)
    return p0, p1, p2, p3, p4, p5
# ----------------------------------------------------------------------
@nbDecC
def make_poly4_exp(y0, dy0, ddy0, Dx, y, dy, ddy):
    Dx2 = Dx * Dx
    Dx3 = Dx * Dx2
    Dx4 = Dx2 * Dx2
    expm1x = np.expm1(Dx)
    A = np.array(((Dx3, Dx4, expm1x - 0.5 * Dx2 - Dx),
                    (3. * Dx2, 4. * Dx3, expm1x - Dx),
                    (6. * Dx, 12. * Dx4, expm1x)))

    b = np.zeros((3, len(y)))

    b[0] = y - (y0 + dy0 * Dx + 0.5 * ddy0 * Dx2)
    b[1] = dy - (ddy0 * Dx + dy0)
    b[2] = ddy - ddy0

    p3, p4, p5 = np.linalg.solve(A, b)

    p0 = y0 - p5
    p1 = dy0 - p5
    p2 = 0.5 * (ddy0 - p5)
    return p0, p1, p2, p3, p4, p5
# ----------------------------------------------------------------------
@nbDecC
def substeps5(fun, x_old, y_old, h, A, B, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev):
    """First substep with fifth degree polynomial."""
    dy_old = K[0, 0]
    ddy_old = K[1, 0]
    Dx = C[1] * h
    if h_prev < 0.:
        p0, p1, p2, p3, p4, p5 = make_poly5(y_old, dy_old, ddy_old,
                                            -h_prev, y_prev,  dy_prev, ddy_prev)

        dy = Dx * (Dx * (Dx * ((5. * Dx) * p5 + 4. * p4) + 3. * p3) + 2. * p2) + p1
        y = Dx * (Dx * (Dx * (Dx * (Dx * p5 + p4) + p3) + p2) + p1) + p0
        # p0, p1, p2, p3, p4, p5 = make_poly4_exp(y_old, dy_old, ddy_old,
        #                                     -h_prev, y_prev,  dy_prev, ddy_prev)
        # Dx2 = Dx * Dx
        # Dx3 = Dx2 * Dx
        # Dx4 = Dx2 * Dx2
        # expx = np.exp(Dx)
        # dy = (p5 * expx
        #     + p4 * (4. * Dx4)
        #     + p3 * (3. * Dx2)
        #     + p2 * (2. * Dx)
        #     + p1)
        # y = p5 * expx + p4 * Dx4 + p3 * Dx3 + p2 * Dx2 + p1 * Dx + p0
    else:
        a = 0.5 * ddy_old
        b = dy_old
        c = y_old

        dy = dy_old + ddy_old * Dx
        y = Dx * (a * Dx + b) + c

    K[0, 1] = dy
    K[1, 1] = fun(x_old + Dx, y, dy)

    for s in range(2, n_stages):
        ah = A[s,:s] * h
        dy = dy_old + np.dot(ah, K[1, :s])
        K[0, s] = dy
        K[1, s] = fun(x_old + C[s] * h, y_old + np.dot(ah, K[0, :s]), dy)
    Bh = B * h

    Ddy = np.dot(Bh, K[1, :-1])
    dy = dy_old + Ddy
    y = y_old + np.dot(Bh, K[0, :-1])

    K[0, -1] = dy
    K[1, -1] = fun(x_old + h, y, dy)
    return y
# ----------------------------------------------------------------------
@nbDecC
def step(fun: ODE2Type,
          direction: np.float64,
          x: np.float64,
          y: npAFloat64,
          y_prev: npAFloat64,
          x_bound: np.float64,
          h_abs: np.float64,
          h: np.float64,
          max_step: np.float64,
          K: npAFloat64,
          n_stages: np.uint64,
          rtol: npAFloat64,
          atol: npAFloat64,
          A: npAFloat64,
          B: npAFloat64,
          C: npAFloat64,
          E: npAFloat64,
          error_exponent: np.float64) -> tuple[bool,
                                          np.float64,
                                          npAFloat64,
                                          np.float64,
                                          np.float64,
                                          np.uint64]:
    if direction * (x - x_bound) >= 0: # x_bound has been reached
        return False, x, y, h_abs, h, np.uint64(0)

    h_abs, x_old, eps, min_step = step_prep(h_abs, x, direction)
    y_old = y
    h_prev = h
    # Saving previous step info
    dy_prev = K[0, 0]
    ddy_prev = K[1, 0]
    # Prepping ofor next step
    K[0, 0] = K[0, -1]
    K[1, 0] = K[1, -1]
    nfev = np.uint64(0)
    while True: # := not working
        x, h, h_abs = h_prep(h_abs, max_step, eps, x_old, x_bound, direction)

        # _RK[1] core loop
        # len(K) == n_stages + 1
        y = substeps3(fun, x_old, y_old, h, A, B, C, K, n_stages, h_prev, y_prev, dy_prev, ddy_prev)
        nfev += n_stages
        # Error calculation
        error_norm2 = (calc_error_norm2(K[0], E, h, y, y_old, rtol, atol)
                       + calc_error_norm2(K[1], E, h, K[0, -1], K[0, 0], rtol, atol)) * 0.5

        if error_norm2 < 1.:
            h_abs *= (MAX_FACTOR if error_norm2 == 0. else
                      min(MAX_FACTOR, SAFETY * error_norm2 ** error_exponent))
            return True, x, y, h_abs, h, nfev # Step is accepted
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm2 ** error_exponent)
            if h_abs < min_step:
                return False, x, y, h_abs, h, nfev # Too small step size
# ----------------------------------------------------------------------
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
                 B: npAFloat64,
                 C: npAFloat64,
                 E: npAFloat64):
        self.n_stages = n_stages
        self.A = A
        self.B = B
        self.C = C
        self.E = E
        self.fun = fun
        self.x = x0
        self.y = y0
        self.y_prev = y0
        self.x_bound = x_bound
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.error_exponent = error_exponent

        self.K = np.zeros((2, self.n_stages + 1, len(y0)), dtype = y0.dtype)
        self.K[0, -1] = dy0
        self.K[1, -1] = self.fun(self.x, self.y, self.dy)
        self._nfev = np.uint64(1)
        self.direction = 1. if x_bound == x0 else np.sign(x_bound - x0)

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.x, y0, dy0, self.K[1, -1], self.direction,
                self.error_exponent, self.rtol, self.atol)
            self._nfev += np.uint64(1)
        else:
            self.h_abs = np.abs(first_step)
        self.step_size = 0.
    # ------------------------------------------------------------------
    def step(self) -> bool:
        y_prev = self.y
        (running,
         self.x,
         self.y,
         self.h_abs,
         self.step_size,
         nfev) = step(self.fun,
                        self.direction,
                        self.x,
                        self.y,
                        self.y_prev,
                        self.x_bound,
                        self.h_abs,
                        self.step_size,
                        self.max_step,
                        self.K,
                        self.n_stages,
                        self.rtol,
                        self.atol,
                        self.A,
                        self.B,
                        self.C,
                        self.E,
                        self.error_exponent)
        self.y_prev = y_prev
        self._nfev += nfev
        return running
    # ------------------------------------------------------------------
    @property
    def dy(self) -> npAFloat64:
        return self.K[-1, 0]
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[np.float64, npAFloat64, npAFloat64]:
        return self.x, self.y, self.K[-1, 0]
# ======================================================================
# @nb.njit(cache = False) # Some issue in making caching jitclasses
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
# ----------------------------------------------------------------------
class RK_Solver2:
    _solver_params: RK_Params
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
                first_step: float = 0.) -> _RK2:
        y0, dy0, rtol, atol = convert(y0, dy0, rtol, atol)
        return init_RK2(fun,
                        np.float64(x0),
                        y0,
                        dy0,
                        np.float64(x_bound),
                        np.float64(max_step),
                        rtol,
                        atol,
                        np.float64(first_step),
                        tuple(cls._solver_params))
# ----------------------------------------------------------------------
class RK23_2(RK_Solver2):
    _solver_params = RK23_params
# ----------------------------------------------------------------------
class RK45_2(RK_Solver2):
    _solver_params = RK45_params
# ----------------------------------------------------------------------
class Solvers2(IterableNamespace):
    RK23_2 = RK23_2
    RK45_2 = RK45_2
