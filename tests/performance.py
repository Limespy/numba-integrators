# type: ignore
from time import perf_counter

import numba as nb
import numpy as np

# ======================================================================
def time_dependent_accuracy() -> float:
    import numba_integrators as ni
    from numba_integrators import reference as ref
    from scipy.integrate import RK45

    x_end = 10
    y_analytical = ref.riccati_solution(x_end)
    kwargs = dict(t_bound = x_end,
                  atol = 1e-10,
                  rtol = 1e-10)
    solver = ni.RK45(ref.riccati_differential, *ref.riccati_initial, **kwargs)
    while ni.step(solver):
        ...
    err_ni = np.abs(solver.y - y_analytical)
    print(f'NI error {err_ni}')
    # ------------------------------------------------------------------
    # Scipy
    solver = RK45(ref.riccati_differential, *ref.riccati_initial, **kwargs)
    while solver.status == 'running':
        solver.step()
    err_scipy = np.abs(solver.y - y_analytical)
    print(f'Scipy error {err_scipy}')

    err_rel = float(err_ni / err_scipy)
    print(f'Relative error to Scipy {err_rel}')
    return err_rel
# ======================================================================
def timing():
    @nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
    def f(t, y):
        return np.array((y[1], -y[0]))

    @nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
    def g(t, y):
        return np.array((2*y[1], -y[0]))

    y0 = np.array((0., 1.))
    t_end = 2000 * np.pi
    args = (f, 0.0, y0)
    kwargs = dict(t_bound = t_end,
                  atol = 1e-10,
                  rtol = 1e-10)
    y_analytical = np.sin(t_end)

    print('numba integrators')

    t0 = perf_counter()
    import numba_integrators as ni
    t_import = perf_counter() - t0
    print(f'Import: {t_import:.2f} s')

    t0 = perf_counter()

    solver = ni.RK45(g, 0.0, y0, **kwargs)
    t_init_first = perf_counter() - t0
    print(f'First initialisation: {t_init_first:.2f} s')
    t0 = perf_counter()

    solver = ni.RK45(*args, **kwargs)
    t_init_second = perf_counter() - t0
    print(f'Second initialisation: {t_init_second:.2f} s')

    t0 = perf_counter()
    ni.step(solver)
    t_step_first = perf_counter() - t0
    print(f'First step: {t_step_first:.2f} s')


    t0 = perf_counter()
    n = 1
    while ni.step(solver):
        n += 1
    runtime = perf_counter() - t0
    print(f'Runtime with {n} steps: {runtime:.3f} s')
    t_step = runtime / n
    print(f'Step overhead {t_step*1e6:.2f} μs')

    err_ni = float(np.sum(np.abs(solver.y[0] - y_analytical)))

    print(f'NI error {err_ni}')
    results = {'time': {'import': t_import,
                        'first initialisation': t_init_first,
                        'second initialisation': t_init_second,
                        'first step': t_step_first,
                        'step': t_step}}
    # ------------------------------------------------------------------
    # Scipy
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

    err_scipy = float(np.sum(np.abs(solver.y[0] - y_analytical)))
    print(f'Scipy error {err_scipy}')
    err_rel = float(err_ni / err_scipy)
    print(f'Relative error {err_rel}')
    return results, err_rel
# ======================================================================
def main():

    results, err_time_dependent = timing()

    import numba_integrators as ni

    results['accuracy'] = {'time_independent': err_time_dependent,
                           'time dependent': time_dependent_accuracy()}

    return ni.__version__, results
