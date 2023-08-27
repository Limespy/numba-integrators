import numba as nb
import numpy as np
import pytest
from numba_integrators import _advanced
from numba_integrators import _API
from numba_integrators import _aux

def test_nbAdvanced_ODE_signature():
    _advanced.nbAdvanced_ODE_signature(nb.float64, nb.float64)
# ----------------------------------------------------------------------
def test_nbAdvanced_initial_step_signature():
    _advanced.nbAdvanced_initial_step_signature(nb.float64,
                                           nb.types.Tuple((nb.float64[:],
                                                           nb.float64))(
                                                            nb.float64,
                                            nb.float64[:],
                                            nb.float64).as_type())
# ======================================================================
def np_array_tuple_compare(t1, t2):
    return all(np.all(a1 == a2) for a1, a2 in zip(t1, t2))
# ----------------------------------------------------------------------
def test_np_array_tuple_compare():
    assert np_array_tuple_compare((np.array((1, 2)), np.array((1, 2))),
                                  (np.array((1, 2)), np.array((1, 2))))
    assert not np_array_tuple_compare((np.array((1, 2)), np.array((1, 2))),
                                      (np.array((1, 2)), np.array((1, 1))))
# ======================================================================
class Test_convert():
    y0 = np.array((1,2,3), dtype = np.float64)
    rtol = np.array((1,1,1), dtype = np.float64)
    atol = np.array((2,2,2), dtype = np.float64)
    def test_correct(self):
        assert np_array_tuple_compare(_aux.convert(self.y0,
                                                   self.rtol,
                                                   self.atol),
                                      (self.y0, self.rtol, self.atol))
    # ------------------------------------------------------------------
    def test_tol_fill(self):
        assert np_array_tuple_compare(_aux.convert(self.y0, 1., 2.),
                                      (self.y0, self.rtol, self.atol))
    # ------------------------------------------------------------------
    def test_y0_from_list(self):
        assert np_array_tuple_compare(_aux.convert(list(self.y0),
                                                   self.rtol,
                                                   self.atol),
                                      (self.y0, self.rtol, self.atol))
    # ------------------------------------------------------------------
    def test_single_value_y0(self):
        assert np_array_tuple_compare(_aux.convert(1, 2, 3),
                                      (np.array((1,), dtype = np.float64),
                                       np.array((2,), dtype = np.float64),
                                       np.array((3,), dtype = np.float64)))
    # ------------------------------------------------------------------
    def test_dtype_change(self):
        assert np_array_tuple_compare(_aux.convert(self.y0.astype(np.int32),
                                                   self.rtol.astype(np.int32),
                                                   self.atol.astype(np.int32)),
                                      (self.y0, self.rtol, self.atol))
