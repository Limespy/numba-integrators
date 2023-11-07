# type: ignore
from collections import defaultdict
from itertools import product
from time import perf_counter

import numba as nb
import numpy as np

# ======================================================================
def accuracy() -> dict[str, dict[str, float]]:
    print('ACCURACY')
    import numba_integrators as ni
    from numba_integrators import reference as ref

    kwargs = dict(atol = 1e-10,
                  rtol = 1e-10)

    accuracy = defaultdict(dict)

    for Solver, (problem_name, problem) in product(ni.ALL,
                                                   ref.problems.items()):
        solver = Solver(problem.differential,
                         problem.x0,
                         problem.y0,
                         problem.x_end,
                         **kwargs)
        while ni.step(solver):
            ...

        err_ni = float(np.sum(np.abs(solver.y - ref.riccati.y_end)))

        y_scipy = ref.scipy_solve(Solver,
                                  problem.differential,
                                  problem.x0,
                                  problem.y0,
                                  problem.x_end,
                                  **kwargs)
        err_scipy = float(np.sum(np.abs(y_scipy - ref.riccati.y_end)))
        accuracy[Solver.__name__][problem_name] = float(err_ni / err_scipy)
    return dict(accuracy)
# ======================================================================
def timing():
    print('TIME')
    times = {}

    # print('\nnumba integrators')

    t0 = perf_counter()
    import numba_integrators as ni
    times['import [s]'] = perf_counter() - t0

    from numba_integrators import reference as ref

    problem = ref.exponential
    x_end = 10
    args = (problem.differential, problem.x0, problem.y0)
    kwargs = dict(t_bound = x_end,
                  first_step = 1e-4,
                  max_step = 1e-4)
    y_analytical = problem.solution(x_end)

    @nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
    def g(t, y):
        return 1.1 * y

    t0 = perf_counter()
    solver = ni.RK45(g, problem.x0, problem.y0, **kwargs)
    times['first initialisation [s]'] = perf_counter() - t0
    t0 = perf_counter()

    solver = ni.RK45(*args, **kwargs)
    times['second initialisation [s]'] = perf_counter() - t0

    t0 = perf_counter()
    ni.step(solver)
    times['first step [s]'] = perf_counter() - t0
    n = 0
    t0 = perf_counter()

    while ni.step(solver):
        n += 1

    runtime = perf_counter() - t0
    print(f'Numba Integrators runtime: {runtime:.3f} s')
    t_step_ni = runtime / n

    # ------------------------------------------------------------------
    # Scipy
    # print('\nScipy')

    t0 = perf_counter()
    from scipy.integrate import RK45
    # print(f'Import: {perf_counter() - t0:.2f} s')

    t0 = perf_counter()

    solver = RK45(*args, **kwargs)
    solver.step()
    # print(f'Initialisation: {perf_counter() - t0:.2f} s')

    t0 = perf_counter()
    n = 0
    while solver.status == 'running':
        solver.step()
        n += 1

    runtime = perf_counter() - t0
    t_step_scipy = runtime / n

    times['step'] = t_step_ni / t_step_scipy

    print(f'Scipy runtime: {runtime:.3f} s')

    for key, value in times.items():
        print(f'{key}: {value:.2f}')

    return times
# ======================================================================
def main():
    results = {'time': timing(),
               'accuracy': accuracy()}

    import numba_integrators as ni

    return ni.__version__, results
