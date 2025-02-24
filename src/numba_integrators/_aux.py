import warnings
from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVarTuple

import numba as nb
import numpy as np
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from typing import Any
    from typing import TypeAlias

    from ._types import nbType
    from ._types import npAFloat64
else:
    Any = TypeAlias = npAFloat64 = nbType = object
# ----------------------------------------------------------------------
warnings.filterwarnings(action = 'ignore',
                        category = nb.errors.NumbaExperimentalFeatureWarning)

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

IS_CACHE = False
IS_NUMBA = False

SMALL_NUMBER = np.spacing(np.float64(0.))
# ----------------------------------------------------------------------
# Signatures
def nbA(dim: int = 1, dtype = nb.float64) -> nbType:
    return nb.types.Array(dtype, dim, 'C')
# ----------------------------------------------------------------------
def nbARO(dim: int = 1, dtype = nb.float64) -> nbType:
    return nb.types.Array(dtype, dim, 'C', readonly = True)
# ======================================================================
# Numba decorators

if IS_NUMBA or TYPE_CHECKING:
    nbDec = nb.njit
else:
    def nbDec(f = None, **__):
        return f if callable(f) else nbDec

nbDecC = nbDec(cache = IS_CACHE)
nbDecFC = nbDec(fastmath = True, cache = IS_CACHE)
# ======================================================================
@nbDec(nb.float64(nbA(1)),
         fastmath = True, cache = IS_CACHE)
def norm(x: npAFloat64) -> np.float64:
    """Compute RMS norm."""
    return (x.dot(x) / x.size)**0.5
# ======================================================================
class IterableNamespaceMeta(type):
    _members: tuple[Any]
    # ------------------------------------------------------------------
    def __subclasses__(self):
        return iter(self._members)
    # ------------------------------------------------------------------
    def __iter__(self):
        return iter(self._members)
# ----------------------------------------------------------------------
class IterableNamespace(metaclass = IterableNamespaceMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._members = tuple(value for key, value in cls.__dict__.items()
                             if key not in ('__module__', '__doc__', '_members'))
# ======================================================================
# Solver types
Y = TypeVarTuple('Y',)
X: TypeAlias = float
class Solver(ABC, Generic[*Y]):
    x: X
    x_bound: X
    y: tuple[*Y]
    step_size: X
    max_step: X
    _nfev: int
    _h_abs_min: X
    _h_next: X
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[X, *Y]:
        return self.x, *self.y
     # ------------------------------------------------------------------
    @property
    def nfev(self) -> np.int64:
        return self._nfev
    # ------------------------------------------------------------------
    def _update_h(self, h_abs: X) -> None:
        eps =  np.spacing(self.x)
        self._h_abs_min = 8. * eps
        h_end = self.x_bound - self.x
        self._h_next = np.copysign((min(max(h_abs, self._h_abs_min),
                                        abs(h_end),
                                        self.max_step - eps)),
                                   h_end)
    # ------------------------------------------------------------------
    @abstractmethod
    def step(self) -> bool:
        ...
    # ------------------------------------------------------------------
    @abstractmethod
    def reboot(self, first_step: np.float64 = np.float64(0.)) -> None:
        ...
# ======================================================================
class SolverFirst(Solver[npAFloat64, npAFloat64]):
    ...
# ----------------------------------------------------------------------
class SolverSecond(Solver[npAFloat64, npAFloat64, npAFloat64]):
    ...
# ----------------------------------------------------------------------
