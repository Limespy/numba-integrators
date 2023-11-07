from time import perf_counter

import numba as nb
import numpy as np
# ======================================================================
def timing():
    print('TIME')
    times = {}
    # setup
    t0 = perf_counter()
    import numba_integrators as ni
    times['import [s]'] = perf_counter() - t0

    from numba_integrators import reference as ref

    problem = ref.exponential
    x_end = 10
    args = (problem.differential, problem.x0, problem.y0)
    # ------------------------------------------------------------------
    # Scipy
    # print('\nScipy')

    t0 = perf_counter()
    from scipy.integrate import RK45
    # print(f'Import: {perf_counter() - t0:.2f} s')

    t0 = perf_counter()

    solver = RK45(*args,
                  t_bound = x_end,
                  first_step = 1e-4,
                  max_step = 1e-4)
    solver.step()
    # print(f'Initialisation: {perf_counter() - t0:.2f} s')

    t0 = perf_counter()
    n = 0
    while solver.status == 'running':
        solver.step()
        n += 1

    runtime = perf_counter() - t0
    t_step_scipy = runtime / n

    print(f'Scipy runtime: {runtime:.3f} s')
    # print('\nnumba integrators')
    # ------------------------------------------------------------------
    # ni


    @nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
    def g(t, y):
        return 1.1 * y

    t0 = perf_counter()
    solver = ni.RK45(g, problem.x0, problem.y0,
                     t_bound = x_end,
                     first_step = 1e-4,
                     max_step = 1e-4)
    times['first initialisation [s]'] = perf_counter() - t0
    t0 = perf_counter()

    solver = ni.RK45(*args,
                     t_bound = x_end,
                     first_step = 1e-4,
                     max_step = 1e-4)
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
    times['step'] = t_step_ni / t_step_scipy
    # ------------------------------------------------------------------
    # ni structref
    sr = ni.sr
    t0 = perf_counter()
    solver = ni.sr.RK45(g, problem.x0, problem.y0,
                     x_bound = x_end,
                     first_step = 1e-4,
                     max_step = 1e-4)
    times['first initialisation sr [s]'] = perf_counter() - t0
    t0 = perf_counter()

    solver = ni.sr.RK45(*args,
                     x_bound = x_end,
                     first_step = 1e-4,
                     max_step = 1e-4)
    times['second initialisation sr [s]'] = perf_counter() - t0

    t0 = perf_counter()
    ni.sr.step(solver)
    times['first step [s]'] = perf_counter() - t0
    n = 0
    t0 = perf_counter()
    while sr.step(solver):
        n += 1

    runtime = perf_counter() - t0
    print(f'Numba Integrators runtime: {runtime:.3f} s')
    t_step_ni_nt = runtime / n

    times['step sr'] = t_step_ni_nt / t_step_scipy
    # ------------------------------------------------------------------
    # printing
    for key, value in times.items():
        if value < 0.1:
            print(f'{key}: {value:.3f}')
        else:
            print(f'{key}: {value:.2f}')

    return times
# ======================================================================
def accuracy() -> dict[str, dict[str, float]]:
    print('ACCURACY')
    import numba_integrators as ni
    from numba_integrators import reference as ref

    kwargs = dict(atol = 1e-10,
                  rtol = 1e-10)

    results = {}

    for Solver in ni.ALL:
        solver_results = {}
        for problem in ref.Problem.problems:
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
            solver_results[problem.name] = float(err_ni / err_scipy)
        results[Solver.__name__] = solver_results
    return results
# ======================================================================
def main():
    results = {'time': timing(),
               'accuracy': accuracy()}

    import numba_integrators as ni

    return ni.__version__, results
