"""API for the package."""
from typing import Any
from typing import Callable

import numba as nb
import numpy as np

from ._aux import Solver
# ======================================================================
# @nb.njit
def step(solver: Solver) -> bool:
    """Taking step with a solver in functional style."""
    return solver.step()
# ======================================================================
# FAST FORWARD
@nb.njit
def ff2x(solver: Solver, x_end: np.float64) -> bool:
    """Fast forwards to given time or x_bound."""
    x_bound = solver.x_bound
    is_nox_last = x_bound > x_end
    if is_nox_last:
        solver.x_bound = x_end
    while solver.step():
        ...

    solver.x_bound = x_bound

    return is_nox_last # type: ignore[return-value]
# ----------------------------------------------------------------------
@nb.njit
def ff2cond(solver: Solver,
            condition: Callable[[Any, Any], bool],
            parameters: Any) -> bool:
    """Fast forwards to given time or x_bound."""
    while solver.step():
        if condition(solver, parameters):
            return True
    return False
