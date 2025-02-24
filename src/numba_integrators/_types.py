from typing import TYPE_CHECKING

from ._aux import PASS_THROUGH
if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Callable
    from typing import Any
    from typing import TypeAlias
    from typing import overload
    from typing import TypeVar
    from typing import ParamSpec
    from typing import Protocol
    from typing import TypeVarTuple

    import numba as nb
    import numpy as np
    from numpy.typing import NDArray
    # ==================================================================
    # Typevar
    T = TypeVar('T')
    P = ParamSpec('P')
    # ==================================================================
    # Numpy types
    npA: TypeAlias = NDArray[Any]
    npAFloat64: TypeAlias = NDArray[np.float64]
    npAInt64: TypeAlias = NDArray[np.int64]

    ODEA_return: TypeAlias = tuple[npAFloat64, Any]

    Arrayable: TypeAlias = int | float | npAFloat64 | Iterable
    # ==================================================================
    # numba types
    nbType: TypeAlias = nb.core.types.abstract.Type
    nbSignature: TypeAlias = nb.core.typing.templates.Signature
# ======================================================================
# copy_type
class Missing: ...

MISSING = Missing()

PASS_THROUGH = lambda _: _
# ----------------------------------------------------------------------
if TYPE_CHECKING:
    @overload
    def copy_type(source: T, /, target: Missing = ...) -> Callable[..., T]: ...
    @overload
    def copy_type(source: T, /, target: Any = ...) -> T: ...
else:
    npA = npAFloat64 = npAInt64 = ODEA_return = Arrayable = object
# ======================================================================
# Copy_type
def copy_type(source, /, target = MISSING):
    return PASS_THROUGH if target is MISSING else target
# ======================================================================
# Solver types

Y = TypeVarTuple('Y',)

class SolverType(Protocol[*Y]):
    x: float
    y: tuple[*Y]
    # ------------------------------------------------------------------
    def step(self) -> bool:
        ...
    # ------------------------------------------------------------------
    @property
    def state(self) -> tuple[float, *Y]:
        return self.x, *self.y
# ======================================================================
class SolverFirstType(SolverType[npAFloat64, npAFloat64]):
    ...
# ----------------------------------------------------------------------
class SolverSecondType(SolverType[npAFloat64, npAFloat64, npAFloat64]):
    ...
# ----------------------------------------------------------------------
