import warnings
from collections.abc import Callable
from collections.abc import Iterable
from math import ulp
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import TypeAlias

import numba as nb
import numpy as np
from numpy.typing import NDArray
# ======================================================================
warnings.filterwarnings(action = 'ignore',
                        category = nb.errors.NumbaExperimentalFeatureWarning)

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

IS_CACHE = True
IS_NUMBA = False

SMALL_NUMBER = np.spacing(np.float64(0.))

# Types
npA: TypeAlias = NDArray[Any]
npAFloat64: TypeAlias = NDArray[np.float64]
npAInt64: TypeAlias = NDArray[np.int64]

_ODEA_return: TypeAlias = tuple[npAFloat64, Any]

ODEA2Type: TypeAlias = Callable[[np.float64, npAFloat64, npAFloat64, Any],
                              _ODEA_return]
Arrayable: TypeAlias = int | float | npAFloat64 | Iterable

# numba types
nbType: TypeAlias = nb.core.types.abstract.Type
nbSignature: TypeAlias = nb.core.typing.templates.Signature
# ----------------------------------------------------------------------
# Signatures
def nbA(dim: int = 1, dtype = nb.float64) -> nbType:
    return nb.types.Array(dtype, dim, 'C')
# ----------------------------------------------------------------------
def nbARO(dim: int = 1, dtype = nb.float64) -> nbType:
    return nb.types.Array(dtype, dim, 'C', readonly = True)
# ======================================================================
# Numba decorators

if IS_NUMBA:
    nbDec = nb.njit
else:
    def nbDec(f = None, **__):
        if callable(f):
            return f
        return nbDec
nbDecC = nbDec(cache = IS_CACHE)
nbDecFC = nbDec(fastmath = True, cache = IS_CACHE)
# ======================================================================
@nbDec(nb.float64(nbA(1)),
         fastmath = True, cache = IS_CACHE)
def norm(x: npAFloat64) -> np.float64:
    """Compute RMS norm."""
    size = x.size
    x *= x
    return (np.sum(x) / size)**0.5
# ======================================================================
# RK_params_type: TypeAlias = tuple[np.float64,
#                                   np.int8,
#                                   npAFloat64,
#                                   npAFloat64,
#                                   npAFloat64,
#                                   npAFloat64]
# ----------------------------------------------------------------------
class RK_Params(NamedTuple):
    n_stages: np.int64
    A: npAFloat64
    C: npAFloat64
    error_exponent: np.float64
# ----------------------------------------------------------------------
def make_rk_params(order: int | float,
                   A: Iterable[Iterable[float]],
                   C: Iterable[float]) -> RK_Params:

    if not C:
        C = [sum(row) for row in A]
    _A = np.array(A, np.float64)
    _C =  np.array(C, np.float64)
    _A[:-2] /= _C.reshape(-1, 1)[1:]
    A_sums = np.sum(_A, axis = 1)
    if not (np.allclose(A_sums[:-1], 1., 5e-16, 5e-16)
            and abs(A_sums[-1]) < 1e-16):
        raise ValueError(A_sums)
    return RK_Params(np.int64(len(C) + 1),
                     _A, _C,
                     np.float64(-0.5 / (order + 1.)),)
# ----------------------------------------------------------------------
RK23_params = make_rk_params(2.,
                 ((0.,  3/4,    0.,  0.),
                  (2/9, 1/3,    4/9,  0.),
                  (5/72, -1/12, -1/9, 1/8)),
                 (1/2, 3/4))
# ----------------------------------------------------------------------
RK45_params = make_rk_params(4.,
((3/40,       9/40,        0.,         0.,       0.,           0.,      0.),
 (44/45,      -56/15,      32/9,       0.,       0.,           0.,      0.),
 (19372/6561, -25360/2187, 64448/6561, -212/729, 0.,           0.,      0.),
 (9017/3168,  -355/33,     46732/5247, 49/176,   -5103/18656,  0.,      0.),
 (35/384,     0.,          500/1113,   125/192,  -2187/6784,   11/84,   0.),
 (-71/57600,  0,           71/16695,   -71/1920, 17253/339200, -22/525, 1/40)),
(1/5, 3/10, 4/5, 8/9, 1.))

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
@nbDecC
def calc_eps(value: float, direction: float) -> float:
    return np.abs(np.nextafter(value, direction * np.inf) - value)
# ======================================================================
@nbDec(nbA(1)(nbA(1), nbARO(1), nbARO(1)), cache = IS_CACHE)
def calc_tolerance(y_abs: npAFloat64, rtol: npAFloat64, atol: npAFloat64
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
class SolverBase:
    _nfev: nb.int64
    x: nb.float64
    x_bound: nb.float64
    h_abs: nb.float64
    _len: nb.float64
    _h_next: nb.float64
    _h_abs_min: nb.float64
    # ------------------------------------------------------------------
    def _calc_h(self, h_abs: np.float64, h_end: np.float64) -> None:
        eps =  np.spacing(self.x)
        self._h_abs_min = 8. * eps
        self._h_next = np.copysign((min(max(h_abs, self._h_abs_min),
                                        abs(h_end),
                                        self.max_step - eps)),
                                   h_end)
    # ------------------------------------------------------------------
    def step(self) -> bool:
        if self._h_next != 0.:
            error, nfev = self._step()
            self._nfev += np.int64(nfev)
            if error < 1.:
                scaler = SAFETY * (error + SMALL_NUMBER) ** self.error_exponent
                self._calc_h(abs(self.step_size) * min(MAX_FACTOR, scaler),
                             self.x_bound - self.x)
                return True
        return False
    # ------------------------------------------------------------------
    def reboot(self, first_step: np.float64 = 0.):
        y_len = len(self.y)
        self._init_K(y_len)
        self._len = 1./y_len
        self.step_size = 0.
        h_end = self.x_bound - self.x

        h_abs, self._nfev = ((np.abs(first_step), np.int64(1))
                                  if first_step else
                                  (self._select_initial_step(h_end),
                                   np.int64(2)))

        self._calc_h(h_abs, h_end)
    # ------------------------------------------------------------------
    def _init_K(self, y_len: int) -> None:
        ...
    # ------------------------------------------------------------------
    def _select_initial_step(self, direction: np.float64) -> np.float64:
        return np.float64(0.)
    # ------------------------------------------------------------------
    def _step(self) -> tuple[np.float64, np.int64]:
        return np.float64(np.inf), np.int64(9)
    # ------------------------------------------------------------------
    @property
    def nfev(self) -> np.int64:
        return self._nfev
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
@nbDecC
def calc_error(step_err_estimator: npAFloat64,
                    y: np.float64,
                    y0_abs: npAFloat64,
                    rtol: npAFloat64,
                    atol: npAFloat64) -> np.float64:
    y_abs = np.abs(y)
    np.maximum(y0_abs, y_abs, y_abs)
    step_err_estimator /= calc_tolerance(y_abs, rtol, atol)
    return np.dot(step_err_estimator, step_err_estimator)
# ======================================================================
@nbDecC
def step_prep(x0: np.float64,
              h_abs: np.float64,
              x_bound: np.float64,
              h_end: np.float64,
              max_step: np.float64
              ) -> tuple[np.float64, np.float64]:
    """Prep for explicit RK solver step."""
    eps = abs(np.nextafter(x0, x_bound) - x0)
    return 8 * eps, min(h_abs, abs(h_end), max_step - eps)
# ======================================================================
base_spec = {'x': nb.float64,
             'y': nbA(1),
             'x_bound': nb.float64,
             'max_step': nb.float64,
             'atol': nbARO(1),
             'rtol': nbARO(1),
             'n_stages': nb.int64,
             'A': nbARO(2),
             'C': nbARO(1),
             'error_exponent': nb.float64,
             'K': nbA(2),
             '_h_next': nb.float64,
             '_h_abs_min': nb.float64,
             'step_size': nb.float64,
             '_nfev': nb.int64,
             '_len': nb.float64}

# ======================================================================
if IS_NUMBA:
    def jitclass_from_dict(update: dict[str, nbType] = {},
                        remove: Iterable[str] = ()
                        ) -> Callable[[type], Any]:
        spec = base_spec.copy()
        if remove:
            for key in remove:
                spec.pop(key, None)
        if update:
            spec |= update
        return nb.experimental.jitclass(spec)
else:
    def jitclass_from_dict(*args, **kwargs):
        return lambda x: x
