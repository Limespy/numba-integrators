import numba as nb
import numpy as np

@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def riccati_differential(x, y):
    '''https://en.wikipedia.org/wiki/Bernoulli_differential_equation#Example
    '''
    return 2 * y/x - x ** 2 * y ** 2
# ----------------------------------------------------------------------
riccati_initial = (1., np.array((1.,), dtype = np.float64))
# ----------------------------------------------------------------------
def riccati_solution(x):
    '''For initial value (1, 1)'''
    return x ** 2 / (x**5 / 5 + 4/5)
