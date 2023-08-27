'''API for the package'''
from typing import Any
from typing import Callable

import numba as nb
import numpy as np

from ._advanced import Advanced
from ._aux import IS_CACHE
from ._aux import npAFloat64
from ._basic import ALL
from ._basic import RK
from ._basic import RK23
from ._basic import RK45
from ._basic import Solver
# ======================================================================
@nb.njit # type: ignore
def step(solver) -> bool:
    return solver.step()
# ======================================================================
# FAST FORWARD
@nb.njit
def ff2t(solver, t_end: np.float64) -> bool:
    '''Fast forwards to given time or t_bound'''
    t_bound = solver.t_bound
    is_not_last = t_bound > t_end
    if is_not_last:
        solver.t_bound = t_end
    while solver.step():
        ...

    solver.t_bound = t_bound

    return is_not_last
# ----------------------------------------------------------------------
@nb.njit
def ff2cond(solver: Any,
            condition: Callable[[Any, Any], bool],
            parameters: Any) -> bool:
    '''Fast forwards to given time or t_bound'''
    while solver.step():
        if condition(solver, parameters):
            return True
    return False
