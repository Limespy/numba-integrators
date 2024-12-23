# pylint: skip-file
"""This file has been generated automatically."""
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import sympy as sp

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

    a = a
    conditions.append((errs(a, c, dc, 6.0, 2.0),))

    if order == 3:
        return conditions

    a2 = a * a
    P1 = b @ a
    conditions.append((errs(a2, c, dc, 12.0, 3.0),
                       errs(P1, c, dc, 24.0, 6.0),))

    if order == 4:
        return conditions

    a3 = a * a2
    aP1 = a * P1
    Q1 = g @ a
    P2 = b @ a2
    bP1 = b @ P1
    conditions.append((errs(a3, c, dc, 20.0, 4.0),
                       errs(aP1, c, dc, 40.0, 8.0),
                       errs(Q1, c, dc, 120.0, 24.0),
                       errs(P2, c, dc, 60.0, 12.0),
                       errs(bP1, c, dc, 120.0, 24.0),))

    if order == 5:
        return conditions

    a4 = a * a3
    a2P1 = a2 * P1
    aQ1 = a * Q1
    aP2 = a * P2
    Q2 = g @ a2
    P3 = b @ a3
    P1_2 = P1 * P1
    abP1 = a * bP1
    gP1 = g @ P1
    baP1 = b @ aP1
    bQ1 = b @ Q1
    bP2 = b @ P2
    bbP1 = b @ bP1
    conditions.append((errs(a4, c, dc, 30.0, 5.0),
                       errs(a2P1, c, dc, 60.0, 10.0),
                       errs(aQ1, c, dc, 180.0, 30.0),
                       errs(aP2, c, dc, 90.0, 15.0),
                       errs(Q2, c, dc, 360.0, 60.0),
                       errs(P3, c, dc, 120.0, 20.0),
                       errs(P1_2, c, dc, 120.0, 20.0),
                       errs(abP1, c, dc, 180.0, 30.0),
                       errs(gP1, c, dc, 720.0, 120.0),
                       errs(baP1, c, dc, 240.0, 40.0),
                       errs(bQ1, c, dc, 720.0, 120.0),
                       errs(bP2, c, dc, 360.0, 60.0),
                       errs(bbP1, c, dc, 720.0, 120.0),))

    if order == 6:
        return conditions

    a5 = a * a4
    a3P1 = a3 * P1
    a2Q1 = a2 * Q1
    a2P2 = a2 * P2
    aQ2 = a * Q2
    aP3 = a * P3
    aP1_2 = a * P1_2
    Q3 = g @ a3
    Q1P1 = Q1 * P1
    P4 = b @ a4
    P2P1 = P2 * P1
    a2bP1 = a2 * bP1
    agP1 = a * gP1
    abaP1 = a * baP1
    abQ1 = a * bQ1
    abP2 = a * bP2
    gaP1 = g @ aP1
    gQ1 = g @ Q1
    gP2 = g @ P2
    ba2P1 = b @ a2P1
    baQ1 = b @ aQ1
    baP2 = b @ aP2
    bQ2 = b @ Q2
    bP3 = b @ P3
    bP1_2 = b @ P1_2
    P1bP1 = P1 * bP1
    abbP1 = a * bbP1
    gbP1 = g @ bP1
    babP1 = b @ abP1
    bgP1 = b @ gP1
    bbaP1 = b @ baP1
    bbQ1 = b @ bQ1
    bbP2 = b @ bP2
    bbbP1 = b @ bbP1
    conditions.append((errs(a5, c, dc, 42.0, 6.0),
                       errs(a3P1, c, dc, 84.0, 12.0),
                       errs(a2Q1, c, dc, 252.0, 36.0),
                       errs(a2P2, c, dc, 126.0, 18.0),
                       errs(aQ2, c, dc, 504.0, 72.0),
                       errs(aP3, c, dc, 168.0, 24.0),
                       errs(aP1_2, c, dc, 168.0, 24.0),
                       errs(Q3, c, dc, 840.0, 120.0),
                       errs(Q1P1, c, dc, 504.0, 72.0),
                       errs(P4, c, dc, 140.0, 20.0),
                       errs(P2P1, c, dc, 252.0, 36.0),
                       errs(a2bP1, c, dc, 252.0, 36.0),
                       errs(agP1, c, dc, 1008.0, 144.0),
                       errs(abaP1, c, dc, 336.0, 48.0),
                       errs(abQ1, c, dc, 1008.0, 144.0),
                       errs(abP2, c, dc, 504.0, 72.0),
                       errs(gaP1, c, dc, 1680.0, 240.0),
                       errs(gQ1, c, dc, 5040.0, 720.0),
                       errs(gP2, c, dc, 2520.0, 360.0),
                       errs(ba2P1, c, dc, 420.0, 60.0),
                       errs(baQ1, c, dc, 1260.0, 180.0),
                       errs(baP2, c, dc, 630.0, 90.0),
                       errs(bQ2, c, dc, 2520.0, 360.0),
                       errs(bP3, c, dc, 840.0, 120.0),
                       errs(bP1_2, c, dc, 840.0, 120.0),
                       errs(P1bP1, c, dc, 504.0, 72.0),
                       errs(abbP1, c, dc, 1008.0, 144.0),
                       errs(gbP1, c, dc, 5040.0, 720.0),
                       errs(babP1, c, dc, 1260.0, 180.0),
                       errs(bgP1, c, dc, 5040.0, 720.0),
                       errs(bbaP1, c, dc, 1680.0, 240.0),
                       errs(bbQ1, c, dc, 5040.0, 720.0),
                       errs(bbP2, c, dc, 2520.0, 360.0),
                       errs(bbbP1, c, dc, 5040.0, 720.0),))

    if order == 7:
        return conditions

    return conditions


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

    a = a
    conditions.append((errs_sp(a, c, dc, 6, 2),))

    if order == 3:
        return conditions

    a2 = a * a
    P1 = b @ a
    conditions.append((errs_sp(a2, c, dc, 12, 3),
                       errs_sp(P1, c, dc, 24, 6),))

    if order == 4:
        return conditions

    a3 = a * a2
    aP1 = a * P1
    Q1 = g @ a
    P2 = b @ a2
    bP1 = b @ P1
    conditions.append((errs_sp(a3, c, dc, 20, 4),
                       errs_sp(aP1, c, dc, 40, 8),
                       errs_sp(Q1, c, dc, 120, 24),
                       errs_sp(P2, c, dc, 60, 12),
                       errs_sp(bP1, c, dc, 120, 24),))

    if order == 5:
        return conditions

    a4 = a * a3
    a2P1 = a2 * P1
    aQ1 = a * Q1
    aP2 = a * P2
    Q2 = g @ a2
    P3 = b @ a3
    P1_2 = P1 * P1
    abP1 = a * bP1
    gP1 = g @ P1
    baP1 = b @ aP1
    bQ1 = b @ Q1
    bP2 = b @ P2
    bbP1 = b @ bP1
    conditions.append((errs_sp(a4, c, dc, 30, 5),
                       errs_sp(a2P1, c, dc, 60, 10),
                       errs_sp(aQ1, c, dc, 180, 30),
                       errs_sp(aP2, c, dc, 90, 15),
                       errs_sp(Q2, c, dc, 360, 60),
                       errs_sp(P3, c, dc, 120, 20),
                       errs_sp(P1_2, c, dc, 120, 20),
                       errs_sp(abP1, c, dc, 180, 30),
                       errs_sp(gP1, c, dc, 720, 120),
                       errs_sp(baP1, c, dc, 240, 40),
                       errs_sp(bQ1, c, dc, 720, 120),
                       errs_sp(bP2, c, dc, 360, 60),
                       errs_sp(bbP1, c, dc, 720, 120),))

    if order == 6:
        return conditions

    a5 = a * a4
    a3P1 = a3 * P1
    a2Q1 = a2 * Q1
    a2P2 = a2 * P2
    aQ2 = a * Q2
    aP3 = a * P3
    aP1_2 = a * P1_2
    Q3 = g @ a3
    Q1P1 = Q1 * P1
    P4 = b @ a4
    P2P1 = P2 * P1
    a2bP1 = a2 * bP1
    agP1 = a * gP1
    abaP1 = a * baP1
    abQ1 = a * bQ1
    abP2 = a * bP2
    gaP1 = g @ aP1
    gQ1 = g @ Q1
    gP2 = g @ P2
    ba2P1 = b @ a2P1
    baQ1 = b @ aQ1
    baP2 = b @ aP2
    bQ2 = b @ Q2
    bP3 = b @ P3
    bP1_2 = b @ P1_2
    P1bP1 = P1 * bP1
    abbP1 = a * bbP1
    gbP1 = g @ bP1
    babP1 = b @ abP1
    bgP1 = b @ gP1
    bbaP1 = b @ baP1
    bbQ1 = b @ bQ1
    bbP2 = b @ bP2
    bbbP1 = b @ bbP1
    conditions.append((errs_sp(a5, c, dc, 42, 6),
                       errs_sp(a3P1, c, dc, 84, 12),
                       errs_sp(a2Q1, c, dc, 252, 36),
                       errs_sp(a2P2, c, dc, 126, 18),
                       errs_sp(aQ2, c, dc, 504, 72),
                       errs_sp(aP3, c, dc, 168, 24),
                       errs_sp(aP1_2, c, dc, 168, 24),
                       errs_sp(Q3, c, dc, 840, 120),
                       errs_sp(Q1P1, c, dc, 504, 72),
                       errs_sp(P4, c, dc, 140, 20),
                       errs_sp(P2P1, c, dc, 252, 36),
                       errs_sp(a2bP1, c, dc, 252, 36),
                       errs_sp(agP1, c, dc, 1008, 144),
                       errs_sp(abaP1, c, dc, 336, 48),
                       errs_sp(abQ1, c, dc, 1008, 144),
                       errs_sp(abP2, c, dc, 504, 72),
                       errs_sp(gaP1, c, dc, 1680, 240),
                       errs_sp(gQ1, c, dc, 5040, 720),
                       errs_sp(gP2, c, dc, 2520, 360),
                       errs_sp(ba2P1, c, dc, 420, 60),
                       errs_sp(baQ1, c, dc, 1260, 180),
                       errs_sp(baP2, c, dc, 630, 90),
                       errs_sp(bQ2, c, dc, 2520, 360),
                       errs_sp(bP3, c, dc, 840, 120),
                       errs_sp(bP1_2, c, dc, 840, 120),
                       errs_sp(P1bP1, c, dc, 504, 72),
                       errs_sp(abbP1, c, dc, 1008, 144),
                       errs_sp(gbP1, c, dc, 5040, 720),
                       errs_sp(babP1, c, dc, 1260, 180),
                       errs_sp(bgP1, c, dc, 5040, 720),
                       errs_sp(bbaP1, c, dc, 1680, 240),
                       errs_sp(bbQ1, c, dc, 5040, 720),
                       errs_sp(bbP2, c, dc, 2520, 360),
                       errs_sp(bbbP1, c, dc, 5040, 720),))

    if order == 7:
        return conditions

    return conditions
