from typing import Any
from typing import Callable

import numba as nb
import numpy as np
from numpy.typing import NDArray
nbtype = nb.core.types.abstract.Type
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
def nbAdvanced_ODE_signature(parameters_type, auxiliary_type):
    return nb.types.Tuple((nb.float64[:],
                           auxiliary_type))(nb.float64,
                                            nb.float64[:],
                                            parameters_type)
# ----------------------------------------------------------------------
def nbAdvanced_initial_step_signature(parameters_type, fun_type):
    return nb.float64(fun_type,
                        nb.float64,
                        nb.float64[:],
                        parameters_type,
                        nb.float64[:],
                        nb.int8,
                        nb.int8,
                        nbARO(1),
                        nbARO(1))
# ----------------------------------------------------------------------
def nbAdvanced_step_signature(parameters_type,
                              auxiliary_type,
                              fun_type):
    return nb.types.Tuple((nb.boolean,
                           nb.float64,
                           nb.float64[:],
                           auxiliary_type,
                           nb.float64,
                           nb.float64,
                           nbA(2)))(fun_type,
                                    nb.int8,
                                    nb.float64,
                                    nb.float64[:],
                                    parameters_type,
                                    nb.float64,
                                    nb.float64,
                                    nb.float64,
                                    nbA(2),
                                    nb.int8,
                                    nbARO(1),
                                    nbARO(1),
                                    nbARO(2),
                                    nbARO(1),
                                    nbARO(1),
                                    nbARO(1),
                                    nb.float64,
                                    auxiliary_type)
# ----------------------------------------------------------------------
def nbA(dim = 1, dtype = nb.float64):
    return nb.types.Array(dtype, dim, 'C')
