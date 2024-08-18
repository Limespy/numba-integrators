import csv
import pathlib
import re
# ======================================================================
PATH_BASE = pathlib.Path(__file__).parent

PATH_RKNF_CONDITIONS_TABLE = PATH_BASE / '_RKNF_condition_table'
PATH_GENERATED = PATH_BASE / '_RKNF_conditions_generated.py'

re_sentinel = re.compile('a|b|g|P|Q')
# ======================================================================
def power(p: int) -> str:
    return '' if p == 1 else str(p)
# ======================================================================
def parse_condition(condition: str):

    if (match := re_sentinel.search(condition[1:])):
        start = 1 + match.start()
        new = condition[:start]
        old = condition[start:]
        return f'{new} @ {old}' if new in ('b', 'g') else f'{new} * {old}'

    if condition[0] == 'a':
        return f'a * a{power(int(p) - 1)}' if (p := condition[1:]) else 'a'

    e, _, p = condition.partition('_')
    return (f'{e} * {e}{'' if p == '2' else f'_{int(p)-1}'}'
            if p else f'{'b' if e[0] == 'P' else 'g'} @ a{power(int(e[1:]))}')
# ======================================================================
HEADER = f'''\
# pylint: skip-file
"""This file has been generated automatically"""
import numba as nb
import numpy as np
import sympy as sp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy import Symbol
    from ..._aux import npAFloat64
    from ._RKNF_conditions import Array
else:
    Symbol = npAFloat64 = Array = None

@nb.njit
def errs(condition: npAFloat64,
         c: npAFloat64,
         dc: npAFloat64,
         div_y: float,
         div_dy: float
         ) -> tuple[float, float]:
    return (abs(c.dot(condition) * div_y - 1.),
            abs(dc.dot(condition) * div_dy - 1.))

def cond2(c: npAFloat64, dc: npAFloat64):
    return (np.sum(c) * 2. - 1., np.sum(dc) - 1.)

def errs_sp(condition: Array,
            c: Array,
            dc: Array,
            div_y: int,
            div_dy: int) -> tuple[Symbol, Symbol]:
    return (c.dot(condition) * div_y - 1,
            dc.dot(condition) * div_dy - 1)
'''

ALL = [HEADER]

conditions = ['''
def conditions(a,
              b,
              g,
              c,
              dc,
              order: int = 0) -> list[tuple[tuple[float, float], ...]]:

    conditions: list[tuple[tuple[float, float], ...]] = []

    conditions.append(((np.sum(c) * 2. - 1., np.sum(dc) - 1.),))

    if order == 2:
        return conditions
''']

conditions_sp = ['''
def conditions_sp(a: Array,
                 b: Array,
                 g: Array,
                 c: Array,
                 dc: Array,
                 order: int = 0
                 ) -> list[tuple[tuple[Symbol, Symbol], ...]]:

    conditions: list[tuple[tuple[Symbol, Symbol], ...]] = []

    conditions.append(((sum(c) * 2 - 1, sum(dc) - 1),))

    if order == 2:
        return conditions
''']


aheader = 'conditions.append(('
apadd = f",\n{' ' * (len(aheader) + 4)}"

# refs_header = 'REFS = ('
# rpadd = f",\n{' ' * (len(refs_header))}"
# refs = ['(0.5, 1.),']

for path in PATH_RKNF_CONDITIONS_TABLE.iterdir():
    _order_i = int(path.stem)
    _order_f = float(_order_i)
    with open(path, newline = '') as f:
        group, divs = zip(*(
            (condition, int(value)) for condition, value
            in csv.reader(f, delimiter = ',')))

    # refs.append(f'{rpadd} '.join((f'({dy * _order}, {dy})' for dy in values_dy)) + ',')

    defs = '\n    '.join((f'{condition} = {parse_condition(condition)}'
                                for condition in group))

    conditions_sp.append(defs)
    conditions.append(defs)
    errs = []
    errs_sp = []
    for var, dy in zip(group, divs):
        dy_f = float(dy)
        errs.append(f'errs({var}, c, dc, {dy_f * _order_f}, {dy_f})')
        errs_sp.append(f'errs_sp({var}, c, dc, {dy * _order_i}, {dy})')

    conditions.append(f'{aheader}{apadd.join(errs)},))\n')
    conditions_sp.append(f'{aheader}{apadd.join(errs_sp)},))\n')

    return_early = f'if order == {_order_i}:\n        return conditions\n'
    conditions.append(return_early)
    conditions_sp.append(return_early)

conditions.append('return conditions\n')
conditions_sp.append('return conditions\n')
ALL.append('\n    '. join(conditions))
ALL.append('\n    '. join(conditions_sp))
# ALL.append(refs_header + rpadd.join((f'({ref})' for ref in refs )) + ')')

PATH_GENERATED.write_text('\n'. join(ALL))
