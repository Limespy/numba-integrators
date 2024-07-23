# pylint: skip-file
"""This file has been generated automatically."""
import numba as nb
from numba.experimental import structref

# Define a StructRef.
# `structref.register` associates the type with the default data model.
# This will also install getters and setters to the fields of
# the StructRef.
@structref.register
class RKType(nb.types.StructRef):
    def preprocess_fields(self, fields):
        # This method is called by the type constructor for additional
        # preprocessing on the fields.
        # Here, we don't want the struct to take Literal types.
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


class RK(structref.StructRefProxy):

    def __new__(cls,
                fun,
                x,
                y,
                rtol,
                atol,
                x_bound,
                max_step,
                h_abs,
                direction,
                step_size,
                nfev,
                error_exponent,
                n_stages,
                A,
                C,
                K):
        return structref.StructRefProxy.__new__(cls,
                                                fun,
                                                x,
                                                y,
                                                rtol,
                                                atol,
                                                x_bound,
                                                max_step,
                                                h_abs,
                                                direction,
                                                step_size,
                                                nfev,
                                                error_exponent,
                                                n_stages,
                                                A,
                                                C,
                                                K)

    @property
    def fun(self):
        return RK_get_fun(self)

    @property
    def x(self):
        return RK_get_x(self)

    @property
    def y(self):
        return RK_get_y(self)

    @property
    def rtol(self):
        return RK_get_rtol(self)

    @property
    def atol(self):
        return RK_get_atol(self)

    @property
    def x_bound(self):
        return RK_get_x_bound(self)

    @x_bound.setter
    def x_bound(self, value):
        RK_set_x_bound(self, value)

    @property
    def max_step(self):
        return RK_get_max_step(self)

    @property
    def h_abs(self):
        return RK_get_h_abs(self)

    @property
    def direction(self):
        return RK_get_direction(self)

    @property
    def step_size(self):
        return RK_get_step_size(self)

    @property
    def nfev(self):
        return RK_get_nfev(self)

    @property
    def error_exponent(self):
        return RK_get_error_exponent(self)

    @property
    def n_stages(self):
        return RK_get_n_stages(self)

    @property
    def A(self):
        return RK_get_A(self)

    @property
    def C(self):
        return RK_get_C(self)

    @property
    def K(self):
        return RK_get_K(self)

    @K.setter
    def K(self, value):
        RK_set_K(self, value)

@nb.njit
def RK_get_fun(self):
    return self.fun

@nb.njit
def RK_get_x(self):
    return self.x

@nb.njit
def RK_get_y(self):
    return self.y

@nb.njit
def RK_get_rtol(self):
    return self.rtol

@nb.njit
def RK_get_atol(self):
    return self.atol

@nb.njit
def RK_get_x_bound(self):
    return self.x_bound

@nb.njit
def RK_set_x_bound(self, value):
    self.x_bound = value

@nb.njit
def RK_get_max_step(self):
    return self.max_step

@nb.njit
def RK_get_h_abs(self):
    return self.h_abs

@nb.njit
def RK_get_direction(self):
    return self.direction

@nb.njit
def RK_get_step_size(self):
    return self.step_size

@nb.njit
def RK_get_nfev(self):
    return self.nfev

@nb.njit
def RK_get_error_exponent(self):
    return self.error_exponent

@nb.njit
def RK_get_n_stages(self):
    return self.n_stages

@nb.njit
def RK_get_A(self):
    return self.A

@nb.njit
def RK_get_C(self):
    return self.C

@nb.njit
def RK_get_K(self):
    return self.K

@nb.njit
def RK_set_K(self, value):
    self.K = value

structref.define_proxy(RK, RKType, ['fun',
                                    'x',
                                    'y',
                                    'rtol',
                                    'atol',
                                    'x_bound',
                                    'max_step',
                                    'h_abs',
                                    'direction',
                                    'step_size',
                                    'nfev',
                                    'error_exponent',
                                    'n_stages',
                                    'A',
                                    'C',
                                    'K'])
