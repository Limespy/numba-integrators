# type: ignore
from time import perf_counter

import numba as nb
import numpy as np

def main():

    @nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
    def f(t, y):
        return np.array((y[1], -y[0]))

    @nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
    def g(t, y):
        return np.array((2*y[1], -y[0]))

    y0 = np.array((0., 1.))

    args = (f, 0.0, y0)
    kwargs = dict(t_bound = 2000 * np.pi,
                  atol = 1e-8,
                  rtol = 1e-8)



    def run_ni():


        print('numba integrators')

        t0 = perf_counter()
        import numba_integrators as ni
        print(f'Import: {perf_counter() - t0:.2f} s')

        t0 = perf_counter()

        solver = ni.RK45(g, 0.0, y0, **kwargs)
        print(f'First initialisation: {perf_counter() - t0:.2f} s')
        t0 = perf_counter()

        solver = ni.RK45(*args, **kwargs)

        print(f'Second initialisation: {perf_counter() - t0:.2f} s')

        t0 = perf_counter()
        solver.step()
        print(f'First step: {perf_counter() - t0:.2f} s')


        t0 = perf_counter()
        n = 1
        while ni.step(solver):#.step():
            n += 1
        runtime = perf_counter() - t0
        print(f'Runtime with {n} steps: {runtime:.3f} s')
        print(f'Step overhead {runtime / n*1e6:.2f} μs')

        err = np.sum(np.abs(solver.y[0] - np.sin(solver.t)))

        print(err)

    def run_scipy():
        print('scipy')

        t0 = perf_counter()
        from scipy.integrate import RK45
        print(f'Import: {perf_counter() - t0:.2f} s')

        t0 = perf_counter()

        solver = RK45(*args, **kwargs)
        solver.step()
        print(f'Initialisation: {perf_counter() - t0:.2f} s')

        t0 = perf_counter()
        n = 1
        while solver.status == "running":
            solver.step()
            n += 1

        runtime = perf_counter() - t0
        print(f'Runtime with {n} steps: {runtime:.3f} s')
        print(f'Step overhead {runtime / n*1e6:.2f} μs')

        err = np.sum(np.abs(solver.y[0] - np.sin(solver.t)))

        print(err)


    for runner in (run_ni, run_scipy):
        runner()
