import numba as nb
import numba_integrators as ni
import numpy as np

@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def f(t, y):
    '''Differential equation for sine wave'''
    return np.array((y[1], -y[0]))

y0 = np.array((0., 1.))

solver = ni.RK45(f, 0.0, y0,
                 t_bound = 1, atol = 1e-8, rtol = 1e-8)

t = []
y = []

while ni.step(solver):
    t.append(solver.t)
    y.append(solver.y)

print(t)
print(y)
