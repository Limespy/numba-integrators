"""API for the package."""
from typing import Any
from typing import Callable

import numba as nb
import numpy as np
from limedev.CLI import get_main

from ._advanced import Advanced
from ._aux import IS_CACHE
from ._aux import npAFloat64
from ._aux import Solver
from ._basic import ALL
from ._basic import RK
from ._basic import RK23
from ._basic import RK45
from ._basic import Solvers
# ======================================================================
@nb.njit
def step(solver: Solver) -> bool:
    """Taking step with a solver in functional style."""
    return solver.step()
# ======================================================================
# FAST FORWARD
@nb.njit
def ff2t(solver: Solver, x_end: np.float64) -> bool:
    """Fast forwards to given time or x_bound."""
    x_bound = solver.x_bound
    is_nox_last = x_bound > x_end
    if is_nox_last:
        solver.x_bound = x_end
    while solver.step():
        ...

    solver.x_bound = x_bound

    return is_nox_last
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
# ======================================================================
main = get_main(__name__)
