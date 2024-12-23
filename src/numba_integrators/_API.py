"""API for the package."""
from typing import TYPE_CHECKING

import numpy as np

from ._aux import nbDec as _nbDec
from ._aux import Solver
from ._aux import SolverFirst
from ._aux import SolverSecond
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    T = TypeVar('T')
else:
    Callable = tuple
    Any = T = object
# ======================================================================
@_nbDec
def step(solver: Solver) -> bool:
    """Taking step with a solver in functional style."""
    return solver.step()
# ======================================================================
# FAST FORWARD
@_nbDec
def ff(solver: Solver, x_end: np.float64 = np.inf) -> bool:
    """Fast forwards to given time or x_bound."""
    x_bound = solver.x_bound
    is_nox_last = x_bound > x_end
    if is_nox_last:
        solver.x_bound = x_end
    while solver.step():
        ...
    solver.x_bound = x_bound

    return is_nox_last and solver.x == x_end # type: ignore[return-value]
# ----------------------------------------------------------------------
@_nbDec
def ff_cond(solver: Solver,
            condition: Callable[[Solver, T], bool],
            parameters: T) -> bool:
    """Fast forwards to given time or x_bound."""
    while solver.step():
        if condition(solver, parameters):
            return True
    return False
