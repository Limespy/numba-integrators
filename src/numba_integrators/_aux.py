import warnings
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Union

import numba as nb
import numpy as np
from numpy.typing import NDArray
warnings.filterwarnings(action='ignore',
                        category = nb.errors.NumbaExperimentalFeatureWarning)

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

IS_CACHE = True

# Types

npAFloat64 = NDArray[np.float64]
npAInt64 = NDArray[np.int64]

ODEFUN  = Callable[[np.float64, npAFloat64], npAFloat64] # type: ignore
ODEFUNA = Callable[[np.float64, npAFloat64, Any], tuple[npAFloat64, Any]] # type: ignore

# numba types
# ----------------------------------------------------------------------
def nbARO(dim = 1, dtype = nb.float64):
    return nb.types.Array(dtype, dim, 'C', readonly = True)
# ----------------------------------------------------------------------
nbODEsignature = nb.float64[:](nb.float64, nb.float64[:])
nbODEtype = nbODEsignature.as_type()

# ----------------------------------------------------------------------
def nbA(dim = 1, dtype = nb.float64):
    return nb.types.Array(dtype, dim, 'C')
# ----------------------------------------------------------------------
@nb.njit(nb.float64(nb.float64[:]),
         fastmath = True, cache = IS_CACHE)
def norm(x: npAFloat64) -> np.float64: # type: ignore
    """Compute RMS norm."""
    return np.sqrt(np.sum(x * x) / x.size) # type: ignore
# ======================================================================
RK23_error_estimator_order = np.int8(2)
RK23_n_stages = np.int8(3)
RK23_A = np.array((
    (0, 0, 0),
    (1/2, 0, 0),
    (0, 3/4, 0)
), dtype = np.float64)
RK23_B = np.array((2/9, 1/3, 4/9), dtype = np.float64)
RK23_C = np.array((0, 1/2, 3/4), dtype = np.float64)
RK23_E = np.array((5/72, -1/12, -1/9, 1/8), dtype = np.float64)
RK23_params  = (RK23_error_estimator_order,
                 RK23_n_stages,
                 RK23_A,
                 RK23_B,
                 RK23_C,
                 RK23_E)
# ----------------------------------------------------------------------
RK45_error_estimator_order = np.int8(4)
RK45_n_stages = np.int8(6)
RK45_A = np.array((
            (0., 0., 0., 0., 0.),
            (1/5, 0., 0., 0., 0.),
            (3/40, 9/40, 0., 0., 0.),
            (44/45, -56/15, 32/9, 0., 0.),
            (19372/6561, -25360/2187, 64448/6561, -212/729, 0),
            (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656)
    ),
                    dtype = np.float64)
RK45_B = np.array((35/384, 0, 500/1113, 125/192, -2187/6784, 11/84),
                   dtype = np.float64)
RK45_C = np.array((0, 1/5, 3/10, 4/5, 8/9, 1), dtype = np.float64)
RK45_E = np.array((-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40),
                   dtype = np.float64)
RK45_params  = (RK45_error_estimator_order,
                 RK45_n_stages,
                 RK45_A,
                 RK45_B,
                 RK45_C,
                 RK45_E)
# ======================================================================
def _into_1d_typearray(item: Union[int, float, npAFloat64, Iterable], # type: ignore
                       length: int = 1,
                       dtype: type = np.float64):
    if isinstance(item, np.ndarray):
        if item.ndim == 0:
            return np.full(length, item, dtype = dtype)
        elif item.ndim == 1:
            return np.asarray(item, dtype = dtype)
        else:
            raise ValueError(f'Dimensionality of y0 is over 1. y0 = {item}')
    elif isinstance(item, Iterable): # Re-checking the item as np array
        return _into_1d_typearray(np.array(item, dtype = dtype),
                                  length,
                                  dtype)
    else:
        return np.full(length, item, dtype = dtype)
# ----------------------------------------------------------------------
def convert(y0: Union[int, float, npAFloat64, Iterable], # type: ignore
            rtol: Union[int, float, npAFloat64, Iterable], # type: ignore
            atol: Union[int, float, npAFloat64, Iterable] # type: ignore
            ) -> tuple[npAFloat64, npAFloat64, npAFloat64]: # type: ignore
    '''Converts y0 and tolerances into correct type of arrays'''
    y0 = _into_1d_typearray(y0)
    return (y0,
            _into_1d_typearray(rtol, len(y0)),
            _into_1d_typearray(atol, len(y0)))
