import numba as nb
import numba_integrators as ni
import numpy as np

@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def f(t, y):
    return np.array((y[1], -y[0]))

y0 = np.array((0., 1.))

args = (f, 0.0, y0)
kwargs = dict(t_bound = 20000 * np.pi,
                atol = 1e-8,
                rtol = 1e-8)

def main():
    solver = ni.RK45(*args, **kwargs)
    while ni.step(solver):#.step():
        ...
