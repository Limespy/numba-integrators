import warnings
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any
from typing import Protocol
from typing import TypeAlias

import numba as nb
import numpy as np
from numpy.typing import NDArray
# ======================================================================
warnings.filterwarnings(action='ignore',
                        category = nb.errors.NumbaExperimentalFeatureWarning)

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

IS_CACHE = True

# Types
npA: TypeAlias = NDArray[Any]
npAFloat64: TypeAlias = NDArray[np.float64]
npAInt64: TypeAlias = NDArray[np.int64]

ODEType: TypeAlias  = Callable[[np.float64, npAFloat64], npAFloat64]
ODEAType: TypeAlias = Callable[[np.float64, npAFloat64, Any],
                              tuple[npAFloat64, Any]]
ODE2Type: TypeAlias  = Callable[[np.float64, npAFloat64, npAFloat64],
                                npAFloat64]
ODEA2Type: TypeAlias = Callable[[np.float64, npAFloat64, npAFloat64, Any],
                              tuple[npAFloat64, Any]]
Arrayable: TypeAlias = int | float | npAFloat64 | Iterable

# numba types
nbType: TypeAlias = nb.core.types.abstract.Type
nbSignature: TypeAlias = nb.core.typing.templates.Signature
# ----------------------------------------------------------------------# Signatures
def nbA(dim: int = 1, dtype = nb.float64) -> nbType:
    return nb.types.Array(dtype, dim, 'C')
# ----------------------------------------------------------------------
def nbARO(dim: int = 1, dtype = nb.float64) -> nbType:
    return nb.types.Array(dtype, dim, 'C', readonly = True)
# ----------------------------------------------------------------------
nbODE_signature = nbA(1)(nb.float64, nbA(1))
nbODE_type = nbODE_signature.as_type()
nbODE2_signature = nbA(1)(nb.float64, nbA(1), nbA(1))
nbODE2_type = nbODE2_signature.as_type()

# ----------------------------------------------------------------------
@nb.njit(nb.float64(nbA(1)),
         fastmath = True, cache = IS_CACHE)
def norm(x: npAFloat64) -> np.float64:
    """Compute RMS norm."""
    size = x.size
    x *= x
    return (np.sum(x) / size)**0.5
# ======================================================================
RK_params_type: TypeAlias = tuple[np.float64,
                                  np.int8,
                                  npAFloat64,
                                  npAFloat64,
                                  npAFloat64,
                                  npAFloat64]
# ----------------------------------------------------------------------
RK23_error_estimator_exponent = np.float64(-1. / (2. + 1.))

RK23_A = np.array((
    (0.,    0.,     0.),
    (1/2,   0.,     0.),
    (0.,    3/4,    0.)
    ), dtype = np.float64)
RK23_B = np.array((2/9, 1/3, 4/9), dtype = np.float64)
RK23_C = np.array((0, 1/2, 3/4), dtype = np.float64)
RK23_E = np.array((5/72, -1/12, -1/9, 1/8), dtype = np.float64)
RK23_n_stages = np.int8(len(RK23_C))
RK23_params: RK_params_type = (RK23_error_estimator_exponent,
                 RK23_n_stages,
                 RK23_A,
                 RK23_B,
                 RK23_C,
                 RK23_E)
# ----------------------------------------------------------------------
RK45_error_estimator_exponent = np.float64(-1. / (4. + 1.))
RK45_A = np.array((
            (0.,        0.,             0.,         0.,         0.),
            (1/5,       0.,             0.,         0.,         0.),
            (3/40,      9/40,           0.,         0.,         0.),
            (44/45,     -56/15,         32/9,       0.,         0.),
            (19372/6561, -25360/2187,   64448/6561, -212/729,   0.),
            (9017/3168, -355/33,        46732/5247, 49/176,     -5103/18656)
    ), dtype = np.float64)
RK45_B = np.array((35/384, 0, 500/1113, 125/192, -2187/6784, 11/84),
                   dtype = np.float64)
RK45_C = np.array((0, 1/5, 3/10, 4/5, 8/9, 1), dtype = np.float64)
RK45_E = np.array((-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40),
                   dtype = np.float64)
RK45_n_stages = np.int8(len(RK45_C))
RK45_params: RK_params_type = (RK45_error_estimator_exponent,
                 RK45_n_stages,
                 RK45_A,
                 RK45_B,
                 RK45_C,
                 RK45_E)
# ======================================================================
def _into_1d_typearray(item: Arrayable,
                       length: int = 1,
                       dtype: type = np.float64) -> npA:
    if isinstance(item, np.ndarray):
        if item.ndim == 0:
            return np.full(length, item, dtype = dtype)
        elif item.ndim == 1:
            return np.asarray(item, dtype = dtype)
        else:
            raise ValueError(f'Dimensionality of y0 is over 1. y0 = {item}')
    if isinstance(item, Iterable): # Re-checking the item as np array
        return _into_1d_typearray(np.array(item, dtype = dtype),
                                  length,
                                  dtype)
    return np.full(length, item, dtype = dtype)
# ----------------------------------------------------------------------
def convert(a: Arrayable, *args: Arrayable
            ) -> tuple[npAFloat64, ...]:
    """Converts y0 and tolerances into correct type of arrays."""
    a = _into_1d_typearray(a)
    length = len(a)
    return a, *(_into_1d_typearray(arg, length) for arg in args)
# ======================================================================
@nb.njit(cache = IS_CACHE)
def calc_eps(value: float, direction: float) -> float:
    return np.abs(np.nextafter(value, direction * np.inf) - value)
# ======================================================================
@nb.njit(nbA(1)(nbA(1), nbARO(1), nbARO(1)), cache = IS_CACHE)
def calc_tolerance(y_abs: npAFloat64,rtol: npAFloat64, atol: npAFloat64
                   ) -> npAFloat64:
    y_abs *= rtol
    y_abs += atol
    return y_abs
# ======================================================================

class Solver(Protocol):
    """Protocol class for basic RK solvers."""
    x: np.float64
    y: npAFloat64
    x_bound: np.float64
    _solver_params: tuple[Any]
    # ------------------------------------------------------------------
    def step(self) -> bool:
        ...
    # ------------------------------------------------------------------
    @property
    def state(self) -> Any:
        ...
# ======================================================================
class IterableNamespaceMeta(type):
    _members: list[Any]
    # ------------------------------------------------------------------
    def __subclasses__(self):
        return iter(self._members)
    # ------------------------------------------------------------------
    def __iter__(self):
        return self.__subclasses__()
# ----------------------------------------------------------------------
class IterableNamespace(metaclass = IterableNamespaceMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        members = [value for key, value in cls.__dict__.items()
                   if key not in ('__module__', '__doc__', '_members')]
        cls._members = members
# ======================================================================
@nb.njit(cache = IS_CACHE)
def calc_error_norm(K: npAFloat64,
                    E: npAFloat64,
                    h: np.float64,
                    y: np.float64,
                    y_old: npAFloat64,
                    rtol: npAFloat64,
                    atol: npAFloat64) -> np.float64:
    step_err_estimator = np.dot(K.T, E)
    step_err_estimator *= h
    y_max_abs = np.abs(y_old)
    np.maximum(y_max_abs, np.abs(y), y_max_abs)
    step_err_estimator /= calc_tolerance(y_max_abs, rtol, atol)
    return norm(step_err_estimator)
# ======================================================================
@nb.njit(cache = IS_CACHE)
def step_prep(h_abs: np.float64, x: np.float64, direction: np.float64):
    """Prep for explicit RK solver step."""
    x_old = x
    eps = calc_eps(x_old, direction)
    min_step = 8 * eps

    if h_abs < min_step:
        h_abs = min_step
    return h_abs, x_old, eps, min_step
# ======================================================================
@nb.njit(cache = IS_CACHE)
def h_prep(h_abs: np.float64,
           max_step: np.float64,
            eps: np.float64,
            x_old: np.float64,
            x_bound: np.float64,
            direction: np.float64) -> tuple[np.float64, np.float64, np.float64]:
    if h_abs > max_step:
        h_abs = max_step - eps
    h = h_abs #* direction
    # Updating
    x = x_old + h

    if direction * (x - x_bound) >= 0: # End reached
        x = x_bound
        h = x - x_old
        h_abs = np.abs(h) # There is something weird going on here
    return x, h, h_abs
# ======================================================================
base_spec = (('A', nbARO(2)),
             ('B', nbARO(1)),
             ('C', nbARO(1)),
             ('E', nbARO(1)),
             ('K', nbA(2)),
             ('n_stages', nb.int8),
             ('x', nb.float64),
             ('y', nbA(1)),
             ('x_bound', nb.float64),
             ('direction', nb.float64),
             ('max_step', nb.float64),
             ('error_exponent', nb.float64),
             ('h_abs', nb.float64),
             ('step_size', nb.float64),
             ('atol', nbARO(1)),
             ('rtol', nbARO(1)))
