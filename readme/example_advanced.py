import numba as nb
import numba_integrators as ni
import numpy as np

@nb.njit
def f(t, y, parameters):
    '''Differential equation for sine wave'''
    auxiliary = parameters[0] * y[1]
    dy = np.array((auxiliary, -y[0])) + parameters[1]
    return dy, auxiliary

t0 = 0.
y0 = np.array((0., 1.))
parameters = (2., np.array((-1., 1.)))

# Numba type signatures
parameters_signature = nb.types.Tuple((nb.float64, nb.float64[:]))
auxiliary_signature = nb.float64
solver_type = ni.RK45

Solver = ni.Advanced(parameters_signature, auxiliary_signature, solver_type)
solver = Solver(f, t0, y0, parameters,
                t_bound = 1, atol = 1e-8, rtol = 1e-8)

t = []
y = []
auxiliary = []

while ni.step(solver):
    t.append(solver.t)
    y.append(solver.y)
    auxiliary.append(solver.auxiliary)

print(t)
print(y)
print(auxiliary)
