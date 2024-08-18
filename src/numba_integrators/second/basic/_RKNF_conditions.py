from itertools import chain

import numpy as np
import sympy as sp
from limedev.CLI import get_main

from ._RKNF_conditions_generated import conditions_sp
# ======================================================================
_SLICE_ALL = slice(None)
# ======================================================================
class Array(sp.Matrix):
    _shape: tuple[int, ...]
    dim: int = 2
    # ------------------------------------------------------------------
    @classmethod
    def _new(cls,  *_, **__):
        obj = super()._new(*_, **__)
        obj._shape = obj.shape
        obj.dim = len(obj._shape)
        return obj
    # ------------------------------------------------------------------
    def __imul__(self, other):
        return self * other
    # ------------------------------------------------------------------
    def __mul__(self, other):
        if isinstance(other, Array):
            return self.multiply_elementwise(other)
        return super().__mul__(other)
    # ------------------------------------------------------------------
    def __matmul__(self, other):
        return super().__mul__(other)
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        dim = self.dim
        if isinstance(key, slice) and dim != 1:
            key = (key, *(_SLICE_ALL for _ in range(dim - 1)))
            item = super().__getitem__(key)
            item._shape = self._shape
            item.dim = self.dim
            return item
        if isinstance(key, tuple):
            key = (*key, *(_SLICE_ALL for _ in range(dim - len(key))))
            item = super().__getitem__(key)
            if any(isinstance(k, slice) for k in key):
                item.shape
                item._shape = (item.shape[1:] if isinstance(key[0], int)
                            else item.shape)

                item.dim = len(item._shape)
            return item
        if isinstance(key, int):
            if dim == 1:
                return self[0, key]
            else:
                # key = (key,) + (_SLICE_ALL,) * (dim - 1)
                # print(self.rows, key)
                item = self.row(key)
                item._shape = self._shape[1:]
                item.dim = len(item._shape)
                return item
        return super().__getitem__(key)
    # ------------------------------------------------------------------
    def __iter__(self):
        if self.dim == 1:
            return (self[0, i] for i in range(self._shape[0]))
        else:
            return (self[index] for index in range(self._shape[0]))
    # ------------------------------------------------------------------
    def __ipow__(self, p):
        return sp.matrices.expressions.HadamardPower(self, p)
    # ------------------------------------------------------------------
    def __pow__(self, p):
        return sp.matrices.expressions.HadamardPower(self, p)
    # ------------------------------------------------------------------
    def squeeze(self):
        ones = tuple(s for s in self.shape if s == 1)
        _shape = tuple(s for s in self.shape if s != 1)
        new_dim = len(_shape)
        if new_dim == 0:
            return self[(0,) * len(ones)]
        new = self.reshape(*(ones + _shape))
        new._shape = _shape
        new.dim = new_dim
        return new
    # ------------------------------------------------------------------
    def dot(self, other):
        return sum(s * o for s, o in zip(self.squeeze(), other.squeeze()))
# ======================================================================
def _make_alpha(steps: int) -> Array:
    def inner(i: int, j: int) -> int | sp.Symbol:
        if i == 0:
            return 0
        if i >= steps - 1:
            return 1
        return sp.Symbol(f'α{i}', real = True, nonnegative = True)

    return Array(steps + 1, 1, inner)
# ----------------------------------------------------------------------
def _make_beta(alpha: Array) -> Array:
    steps = len(alpha) - 1
    def _indicator(i: int, j: int) -> int | sp.Symbol:
        return 0 if (i - j <= 0 or i == 0) else sp.Symbol(f'β{i}{j}',
                                                          real = True)
    beta = Array(steps + 1, steps, _indicator)

    beta[1, 0] = alpha[1, 0]
    for i in range(2, steps + 1):
        beta[i, 0] = alpha[i, 0] - sum(beta[i, 1:])

    return beta
# ----------------------------------------------------------------------
def _make_gamma(alpha: Array) -> Array:
    steps = len(alpha) - 1
    def _indicator(i: int, j: int) -> int | sp.Symbol:
        return 0 if (i - j <= 0 or i == 0 or j == 0) else sp.Symbol(f'γ{i}{j}', real = True)

    gamma = Array(steps + 1, steps, _indicator)

    gamma[1, 0] = alpha[1, 0]**2 / 2
    for i in range(2, steps + 1):
        gamma[i, 0] = alpha[i, 0]**2 / 2 - sum(gamma[i, 1:])
    return gamma
# ======================================================================
def make(steps: int, order: int) -> None:
    alpha = _make_alpha(steps)
    one = sp.S(1)
    alpha[1, 0] = one / 5
    alpha[2, 0] = 2 * one / 5
    alpha[3, 0] = 3 * one / 5
    alpha[4, 0] = 4 * one / 5
    beta = _make_beta(alpha)
    gamma =  _make_gamma(alpha)

    print(repr(alpha))
    print(repr(beta))
    print(repr(gamma))

    conditions = conditions_sp(alpha[:-1], beta[:-1], gamma[:-1], gamma[-1], beta[-1], order)

    y_lin  = [item for item, _ in chain(*conditions[:-1])]
    dy_lin = [item for _, item in chain(*conditions)]

    # solution = sp.solve(y_lin + dy_lin)
    # for row in solution:
    #     for key, value in row.items():
    #         print(key, value)
    # return solution
# ======================================================================
main = get_main(__name__)
# ======================================================================
if __name__ == '__main__':
    raise SystemExit(main())
