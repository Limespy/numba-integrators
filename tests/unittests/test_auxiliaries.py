import numba as nb
import pytest
from numba_integrators import _API

def test_nbAdvanced_ODE_signature():
    _API.nbAdvanced_ODE_signature(nb.float64, nb.float64)

def test_nbAdvanced_initial_step_signature():
    _API.nbAdvanced_initial_step_signature(nb.float64,
                                           nb.types.Tuple((nb.float64[:],
                                                           nb.float64))(
                                                            nb.float64,
                                            nb.float64[:],
                                            nb.float64).as_type())
