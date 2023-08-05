from enum import Enum

import numba as nb
import numpy as np

from ._aux import Float64Array
from ._aux import nbODEtype
from ._aux import nbRO

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

@nb.njit
def norm(x):
    """Compute RMS norm."""
    return np.linalg.norm(x) / x.size ** 0.5

@nb.njit
def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    if y0.size == 0:
        return np.inf

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)

base_spec = (('A', nbRO(2)),
             ('B', nbRO(1)),
             ('C', nbRO(1)),
             ('E', nbRO(1)),
             ('K', nb.types.Array(nb.float64, 2, 'C')),
             ('order', nb.int8),
             ('error_estimator_order', nb.int8),
             ('n_stages', nb.int8),
             ('t_old', nb.float64),
             ('t', nb.float64),
             ('y', nb.float64[:]),
             ('y_old', nb.float64[:]),
             ('t_bound', nb.float64),
             ('direction', nb.int8),
             ('max_step', nb.float64),
             ('error_exponent', nb.float64),
             ('step_size', nb.float64),
             ('h_abs', nb.float64),
             ('fun', nbODEtype),
             ('atol', nb.float64[:]),
             ('rtol', nb.float64[:])
)

@nb.experimental.jitclass(base_spec)
class RungeKutta:
    """Base class for explicit Runge-Kutta methods."""

    def __init__(self, order, error_estimator_order, n_stages, A, B, C, E,
                 fun, t0, y0, t_bound, max_step = np.inf,
                 rtol=1e-3, atol=1e-6, first_step = 0.):
        self.order = order
        self.error_estimator_order = error_estimator_order
        self.n_stages = n_stages
        self.A = A
        self.B = B
        self.C = C
        self.E = E
        self.fun = fun
        self.t = t0
        self.t_old = t0
        self.y = y0
        self.y_old = y0
        self.t_bound = t_bound
        self.K = np.empty((self.n_stages + 1, len(y0)), dtype = self.y.dtype)
        self.K[-1] = self.fun(self.t, self.y)
        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.step_size = 0

        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step

        if not first_step:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, self.K[-1], self.direction,
                self.error_estimator_order, self.rtol, self.atol)
    # ------------------------------------------------------------------
    def _step(self, rejected):

        if self.h_abs > self.max_step:
            self.h_abs = self.max_step
        h = self.h_abs * self.direction
        # Updating
        self.t_old = self.t
        self.t += h
        self.K[0] = self.K[-1]

        if self.direction * (self.t - self.t_bound) > 0:
            self.t = self.t_bound
            h = self.t - self.t_old
            self.h_abs = np.abs(h) # There is something weird going on here

        # RK core loop
        for s in range(1, self.n_stages):
            self.K[s] = self.fun(self.t_old + self.C[s] * h,
                                 self.y + np.dot(self.K[:s].T, self.A[s,:s]) * h)
        # Updating
        self.y_old = self.y

        self.y = self.y_old + h * np.dot(self.K[:-1].T, self.B)

        self.K[-1] = self.fun(self.t + h, self.y)
        x = np.dot(self.K.T, self.E) * h / (self.atol +
                                            np.maximum(np.abs(self.y_old),
                                                       np.abs(self.y)) * self.rtol)

        error_norm = np.sqrt(np.sum(x * x) / x.size)
        if error_norm < 1:

            factor = (MAX_FACTOR if error_norm == 0 else
                      min(MAX_FACTOR,
                          SAFETY * error_norm ** self.error_exponent))
            if rejected and factor > 1:
                factor = 1

            self.h_abs *= factor
            return False
        else:
            self.h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** self.error_exponent)
            return True
    # ------------------------------------------------------------------
    def step(self):
        min_step = 10 * np.abs(np.nextafter(self.t, self.direction * np.inf) - self.t)

        if self.h_abs > self.max_step:
            self.h_abs = self.max_step
        elif self.h_abs < min_step:
            self.h_abs = min_step

        rejected = self._step(False)
        while rejected: # := not working
            if self.h_abs < min_step:
                return False
            rejected = self._step(rejected)

        return self.direction * (self.t - self.t_bound) < 0

# class _RK23(RungeKutta):
#     """Explicit Runge-Kutta method of order 3(2).

#     This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
#     assuming accuracy of the second-order method, but steps are taken using the
#     third-order accurate formula (local extrapolation is done). A cubic Hermite
#     polynomial is used for the dense output.

#     Can be applied in the complex domain.

#     Parameters
#     ----------
#     fun : callable
#         Right-hand side of the system. The calling signature is ``fun(t, y)``.
#         Here ``t`` is a scalar and there are two options for ndarray ``y``.
#         It can either have shape (n,), then ``fun`` must return array_like with
#         shape (n,). Or alternatively it can have shape (n, k), then ``fun``
#         must return array_like with shape (n, k), i.e. each column
#         corresponds to a single column in ``y``. The choice between the two
#         options is determined by `vectorized` argument (see below).
#     t0 : float
#         Initial time.
#     y0 : array_like, shape (n,)
#         Initial state.
#     t_bound : float
#         Boundary time - the integration won't continue beyond it. It also
#         determines the direction of the integration.
#     first_step : float or None, optional
#         Initial step size. Default is ``None`` which means that the algorithm
#         should choose.
#     max_step : float, optional
#         Maximum allowed step size. Default is np.inf, i.e., the step size is not
#         bounded and determined solely by the solver.
#     rtol, atol : float and array_like, optional
#         Relative and absolute tolerances. The solver keeps the local error
#         estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
#         relative accuracy (number of correct digits), while `atol` controls
#         absolute accuracy (number of correct decimal places). To achieve the
#         desired `rtol`, set `atol` to be smaller than the smallest value that
#         can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
#         allowable error. If `atol` is larger than ``rtol * abs(y)`` the
#         number of correct digits is not guaranteed. Conversely, to achieve the
#         desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
#         than `atol`. If components of y have different scales, it might be
#         beneficial to set different `atol` values for different components by
#         passing array_like with shape (n,) for `atol`. Default values are
#         1e-3 for `rtol` and 1e-6 for `atol`.
#     vectorized : bool, optional
#         Whether `fun` is implemented in a vectorized fashion. Default is False.

#     Attributes
#     ----------
#     n : int
#         Number of equations.
#     status : string
#         Current status of the solver: 'running', 'finished' or 'failed'.
#     t_bound : float
#         Boundary time.
#     direction : float
#         Integration direction: +1 or -1.
#     t : float
#         Current time.
#     y : ndarray
#         Current state.
#     t_old : float
#         Previous time. None if no steps were made yet.
#     step_size : float
#         Size of the last successful step. None if no steps were made yet.
#     nfev : int
#         Number evaluations of the system's right-hand side.
#     njev : int
#         Number of evaluations of the Jacobian. Is always 0 for this solver as it does not use the Jacobian.
#     nlu : int
#         Number of LU decompositions. Is always 0 for this solver.

#     References
#     ----------
#     .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
#            Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
#     """
#     order = 3
#     error_estimator_order = 2
#     n_stages = 3
#     C = np.array((0, 1/2, 3/4))
#     A = np.array((
#         (0, 0, 0),
#         (1/2, 0, 0),
#         (0, 3/4, 0)
#     ))
#     B = np.array((2/9, 1/3, 4/9))
#     E = np.array((5/72, -1/12, -1/9, 1/8))
#     P = np.array(((1, -4 / 3, 5 / 9),
#                   (0, 1, -2/3),
#                   (0, 4/3, -8/9),
#                   (0, -1, 1)))

RK45params = dict(
    order = 5,
    error_estimator_order = 4,
    n_stages = 6,
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ]),
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]),
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1]),
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                1/40])
)

def convert(y0, rtol, atol) -> tuple[Float64Array, Float64Array, Float64Array]:
    y0 = np.asarray(y0).astype(np.float64)

    if not isinstance(atol, np.ndarray):
        atol = np.full(len(y0), atol)

    if not isinstance(rtol, np.ndarray):
        rtol = np.full(len(y0), rtol)
    return y0, rtol, atol

def RK45(fun, t0, y0, t_bound, max_step = np.inf,
         rtol=1e-3, atol=1e-6, first_step = 0.):

    y0, rtol, atol = convert(y0, rtol, atol)

    return RungeKutta(*RK45params.values(), # type: ignore
                 fun, t0, y0, t_bound, max_step, rtol, atol, first_step)
