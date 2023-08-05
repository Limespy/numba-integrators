# type: ignore
from time import perf_counter

import numba as nb
import numba_integrators as ni
import numpy as np
from scipy.integrate import RK45

def main():

    @nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
    def f(t, y):
        return np.array((y[1], -y[0]))



    y0 = np.array((0., 1.))

    def run_ni():
        print('numba integrators')

        t0 = perf_counter()

        solver = ni.RK45(f, 0.0, y0, t_bound = 2000 * np.pi,
                    atol = 1e-8,
                    rtol = 1e-8)
        solver.step()
        print(perf_counter() - t0)


        t0 = perf_counter()

        while solver.step():
            ...
        print(perf_counter() - t0)

        err = np.sum(np.abs(solver.y[0] - np.sin(solver.t)))

        print(err)

    def run_scipy():
        print('scipy')

        t0 = perf_counter()

        solver = RK45(f, 0.0, y0, t_bound = 2000 * np.pi,
                    atol = 1e-8,
                    rtol = 1e-8)
        solver.step()
        print(perf_counter() - t0)

        t0 = perf_counter()

        while solver.status == "running":
            solver.step()

        print(perf_counter() - t0)

        err = np.sum(np.abs(solver.y[0] - np.sin(solver.t)))

        print(err)


    for runner in (run_scipy, run_ni,):
        runner()
