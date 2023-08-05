import numba as nb
import numpy as np
from numpy.typing import NDArray

# Types

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]

# numba types

def nbARO(dim = 1, dtype = nb.float64):
    return nb.types.Array(dtype, dim, 'C', readonly = True)

nbODEtype = nb.float64[:](nb.float64, nb.float64[:]).as_type()

def nbA(dim = 1, dtype = nb.float64):
    return nb.types.Array(dtype, dim, 'C')
