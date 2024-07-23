from time import perf_counter

import numba as nb
import numpy as np
from limedev.test import BenchmarkResultsType
from limedev.test import eng_round
from limedev.test import sigfig_round
from limedev.test import YAMLSafe
# ======================================================================
def timing():
    print('TIME')
    times: dict[str, YAMLSafe] = {}
    times['base'] = {}
    # setup
    t0 = perf_counter()
    import numba_integrators as ni
    rounded, prefix = eng_round(perf_counter() - t0)
    times['base'][f'import [{prefix}s]'] = rounded

    from numba_integrators.first import reference as ref

    problem = ref.sine
    x_end = problem.x_end
    args = (problem.differential, problem.x0, problem.y0)

    problem.differential(problem.x0, problem.y0)
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
    nfev_scipy =  solver.nfev
    t_step_scipy = runtime / n
    times['scipy'] = nfev_scipy
    print(f'Scipy runtime: {runtime:.3f} s')
    # ------------------------------------------------------------------
    # ni
    times['jitclass'] = {}

    t0 = perf_counter()

    from numba_integrators.first.basic import RK45

    rounded, prefix = eng_round(perf_counter() - t0)
    times['jitclass'][f'import [{prefix}s]'] = rounded

    @nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
    def g(t, y):
        return 1.1 * y

    g(problem.x0, problem.y0)

    t0 = perf_counter()
    solver = RK45(g, problem.x0, problem.y0,
                     x_bound = x_end,
                     first_step = 1e-4,
                     max_step = 1e-4)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['jitclass'][f'first initialisation [{prefix}s]'] = rounded
    t0 = perf_counter()

    solver = RK45(*args,
                     x_bound = x_end,
                     first_step = 1e-4,
                     max_step = 1e-4)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['jitclass'][f'second initialisation [{prefix}s]'] = rounded

    t0 = perf_counter()
    ni.step(solver)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['jitclass'][f'first step [{prefix}s]'] = rounded
    n = 0
    t0 = perf_counter()

    while ni.step(solver):
        n += 1

    runtime = perf_counter() - t0
    print(f'Numba Integrators runtime: {runtime:.3f} s')
    t_step_ni = runtime / n
    times['jitclass']['step'] = round(t_step_ni / t_step_scipy, 4)
    times['jitclass']['nfev'] = solver.nfev
    # ------------------------------------------------------------------
    # ni structref

    times['structref'] = {}

    t0 = perf_counter()

    from numba_integrators.first import structref as sr

    rounded, prefix = eng_round(perf_counter() - t0)
    times['structref'][f'import [{prefix}s]'] = rounded

    t0 = perf_counter()
    solver = sr.RK45(g, problem.x0, problem.y0,
                     x_bound = x_end,
                     first_step = 1e-4,
                     max_step = 1e-4)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['structref'][f'first initialisation sr [{prefix}s]'] = rounded
    t0 = perf_counter()

    solver = sr.RK45(*args,
                     x_bound = x_end,
                     first_step = 1e-4,
                     max_step = 1e-4)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['structref'][f'second initialisation sr [{prefix}s]'] = rounded

    t0 = perf_counter()
    sr.step(solver)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['structref'][f'first step [{prefix}s]'] = rounded
    n = 0
    t0 = perf_counter()
    while sr.step(solver):
        n += 1

    runtime = perf_counter() - t0
    print(f'Numba Integrators structref runtime: {runtime:.3f} s')
    t_step_ni_nt = runtime / n

    times['structref']['step'] = round(t_step_ni_nt / t_step_scipy, 4)
    times['structref']['nfev'] = solver.nfev
    # ------------------------------------------------------------------
    # # ni second order

    times['1st order'] = {}
    solver = RK45(*args,
                     x_bound = problem.x_end*500.,
                     rtol = 1e-10,
                     atol = 1e-10)
    n = 0
    t0 = perf_counter()
    while ni.step(solver):
        n += 1

    runtime = perf_counter() - t0
    times['1st order']['step'] = round(runtime / n / t_step_scipy, 4)

    print(f'Numba Integrators first order runtime: {runtime:.3f} s')
    times['1st order']['nfev'] = solver.nfev

    times['2nd order'] = {}


    t0 = perf_counter()

    from numba_integrators.second.basic.RK import RK45_2

    rounded, prefix = eng_round(perf_counter() - t0)
    times['2nd order'][f'import [{prefix}s]'] = rounded

    from numba_integrators.second.basic.reference import sine2 as problem2
    problem2.differential(problem2.x0, problem2.y0, problem2.dy0)

    @nb.njit(nb.float64[:](nb.float64, nb.float64[:], nb.float64[:]))
    def g2(t, y, dy):
        return 1.1 * y

    g2(problem2.x0, problem2.y0, problem2.dy0)

    t0 = perf_counter()
    solver = RK45_2(g2,
                    problem2.x0,
                    problem2.y0,
                    problem2.dy0,
                    problem2.x_end,
                    rtol = 1e-10,
                    atol = 1e-10)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['2nd order'][f'first initialisation [{prefix}s]'] = rounded

    @nb.njit(nb.float64[:](nb.float64, nb.float64[:], nb.float64[:]))
    def g2(t, y, dy):
        return 1.2 * y

    g2(problem2.x0, problem2.y0, problem2.dy0)

    t0 = perf_counter()
    solver = RK45_2(g2,
                    problem2.x0,
                    problem2.y0,
                    problem2.dy0,
                    problem2.x_end,
                    rtol = 1e-10,
                    atol = 1e-10)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['2nd order'][f'second initialisation [{prefix}s]'] = rounded

    t0 = perf_counter()
    solver.fun = problem2.differential
    solver.x = np.float64(problem2.x0)
    solver.y = np.array(problem2.y0)
    solver.dy = np.array(problem2.dy0)
    solver.x_bound = np.float64(problem2.x_end*500.)
    solver.reboot()

    rounded, prefix = eng_round(perf_counter() - t0)
    times['2nd order'][f'swapping [{prefix}s]'] = rounded

    t0 = perf_counter()
    ni.step(solver)
    rounded, prefix = eng_round(perf_counter() - t0)
    times['2nd order'][f'first step [{prefix}s]'] = rounded

    n = 0
    t0 = perf_counter()
    while ni.step(solver):
        n += 1
    runtime = perf_counter() - t0

    print(f'Numba Integrators second order runtime: {runtime:.3f} s')

    times['2nd order']['nfev'] = solver.nfev
    times['2nd order']['step'] = round(runtime / n / t_step_scipy, 4)

    return times
# ======================================================================
def accuracy() -> dict[str, dict[str, float]]:
    print('ACCURACY')
    import numba_integrators as ni
    from numba_integrators.first.basic import Solvers
    from numba_integrators.first import reference as ref

    kwargs = dict(atol = 1e-10,
                  rtol = 1e-10)

    results = {}

    for Solver in Solvers:
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
            rel_err = float(err_ni / err_scipy) - 1.
            if rel_err != 0.:
                rel_err = float(np.sign(rel_err)) * sigfig_round(abs(rel_err), 3)
            solver_results[problem.name] = rel_err
        results[Solver.__name__] = solver_results
    return results
# ======================================================================
def main() -> BenchmarkResultsType:
    results = {'time': timing(),
               'accuracy': accuracy()}

    import numba_integrators as ni

    return ni.__version__, results
