from collections.abc import Callable
from typing import Any
from typing import Literal as L

import numpy as np
from limedev.CLI import get_main
from numba_integrators._lnumpy import F64Array
# ======================================================================
def minimise[T](x1: float, e1: float,
             f: Callable[[float], tuple[float, T]],
             df: Callable[[float], float],
             tol: float) -> tuple[float, T]:
    x2 = x1 - e1 / df(x1)
    e2, out = f(x2)
    Delta_e = (e2 - e1)

    while Delta_e / e2 > tol:
        Rp = Delta_e / df(x2)
        x1, x2 = x2, ((x2 * x2 - x1 * x1) * 0.5 - Rp * x2)/(x2 - x1 - Rp)

        e1 = e2
        e2, out = f(x2)
        Delta_e = (e2 - e1)
    return e2, out
# ======================================================================
def newton_step[N_Vars: int,
                N_Vars2: int](Dx: float,
                              err: F64Array[N_Vars2],
                              jac_ddy: F64Array[N_Vars, N_Vars2],
                              f_jac_g: F64Array[N_Vars2, N_Vars2]):
    # print('jac_ddy\n', jac_ddy)
    # Calculate error diffs jacobian
    jac_g = f_jac_g(Dx, jac_ddy)
    # Solve error diffs step
    a = -np.linalg.solve(jac_g, err)
    return a
# ======================================================================
def secant_step(y1, y2, dy1, dy2, err1, err2):
    n = len(y2)
    err_ratio = err2 / (err1 - err2)
    return (y2 - y1) * err_ratio[:n], (dy2 - dy1) * err_ratio[n:]
# ======================================================================
def check_error(err, y, dy, atol: float, rtol: float, n: int) -> bool:
    return (np.any(np.abs(err[:n]) > (atol + rtol * np.abs(y)))
           or np.any(np.abs(err[n:]) > (atol + rtol * np.abs(dy))))
# ======================================================================
def newton_newton(Dx, xi, y_e, dy_e,
                  f_step, f_ddy, f_jac_ddy, f_jac_g, atol, rtol, n):
    # First step with newton iteration
    ddy = f_ddy(xi, y_e, dy_e)

    y_calc, dy_calc = f_step(y_e, dy_e, ddy)

    err_e = np.hstack((y_e - y_calc, dy_e - dy_calc))

    ay, ady = newton_step(Dx, xi, y_e, dy_e, err_e, f_jac_ddy, f_jac_g)
    y1 = y_e
    y2 = y_e + ay
    dy1 = dy_e
    dy2 = dy_e + ady
    ddy = f_ddy(xi, y2, dy2)
    y_calc, dy_calc = f_step(y2, dy2, ddy)

    err1 = err_e
    err2 = np.hstack((y2 - y_calc, dy2 - dy_calc))

    iterations = 1
    print('internal error\n', err2)
    while check_error(err2, y2, dy2, atol, rtol, n):
        # calculate ddy

        a = newton_step(Dx, xi, y2, dy2, err2, f_jac_ddy, f_jac_g)
        y1 = y2
        y2 = y1 + ay
        dy1 = dy2
        dy2 = dy2 + ady
        ddy = f_ddy(xi, y2, dy2)
        y_calc, dy_calc = f_step(y2, dy2, ddy)

        err1 = err2
        err2 = np.hstack((y2 - y_calc, dy2 - dy_calc))
        # dyddy = np.vstack((dy, ddy))
        # # print(dyddy.shape)
        # dddy = (jac_ddy @ dyddy).flatten()[0]
        # # Calculate error
        # print(dddy / FX[3] - 1.)

        print('internal error\n', err2)
        iterations += 1
    print(iterations)
    return y2, dy2, ddy
# ======================================================================
def newton_secant(Dx, xi, y_e, dy_e,
                  f_step, f_ddy, f_jac_ddy, f_jac_g, atol, rtol, n):
    # First step with newton iteration
    ddy = f_ddy(xi, y_e, dy_e)

    y_calc, dy_calc = f_step(y_e, dy_e, ddy)

    err_e = np.hstack((y_e - y_calc, dy_e - dy_calc))

    ay, ady = newton_step(Dx, xi, y_e, dy_e, err_e, f_jac_ddy, f_jac_g)
    y1 = y_e
    y2 = y_e + ay
    dy1 = dy_e
    dy2 = dy_e + ady
    ddy = f_ddy(xi, y2, dy2)
    y_calc, dy_calc = f_step(y2, dy2, ddy)

    err1 = err_e
    err2 = np.hstack((y2 - y_calc, dy2 - dy_calc))

    iterations = 1

    while check_error(err2, y2, dy2, atol, rtol, n):
        # calculate ddy

        ay, ady = secant_step(y1, y2, dy1, dy2, err1, err2)
        y1 = y2
        y2 = y1 + ay
        dy1 = dy2
        dy2 = dy2 + ady
        ddy = f_ddy(xi, y2, dy2)
        y_calc, dy_calc = f_step(y2, dy2, ddy)

        err1 = err2
        err2 = np.hstack((y2 - y_calc, dy2 - dy_calc))
        # dyddy = np.vstack((dy, ddy))
        # # print(dyddy.shape)
        # dddy = (jac_ddy @ dyddy).flatten()[0]
        # # Calculate error
        # print(dddy / FX[3] - 1.)

        print('internal error\n', err2)
        iterations += 1
    print(iterations)
    return y2, dy2, ddy
# ======================================================================
def fixed_secant(Dx, xi, y_e, dy_e,
                 f_step, f_ddy, f_jac_ddy, f_jac_g, atol, rtol, n):
    # First step with newton iteration

    ddy = f_ddy(xi, y_e, dy_e)

    y_calc, dy_calc = f_step(y_e, dy_e, ddy)

    err_e = np.hstack((y_e - y_calc, dy_e - dy_calc))

    y1 = y_e
    y2 = (y_e + y_calc) * 0.5
    y2 = y_calc
    dy1 = dy_e
    dy2 = (dy_e + dy_calc) * 0.5
    dy2 = dy_calc
    ddy = f_ddy(xi, y2, dy2)
    y_calc, dy_calc = f_step(y2, dy2, ddy)

    err1 = err_e
    err2 = np.hstack((y2 - y_calc, dy2 - dy_calc))

    iterations = 1

    while check_error(err2, y2, dy2, atol, rtol, n):
        # calculate ddy

        ay, ady = secant_step(y1, y2, dy1, dy2, err1, err2)
        y1 = y2
        y2 = y1 + ay
        dy1 = dy2
        dy2 = dy2 + ady
        ddy = f_ddy(xi, y2, dy2)
        y_calc, dy_calc = f_step(y2, dy2, ddy)

        err1 = err2
        err2 = np.hstack((y2 - y_calc, dy2 - dy_calc))
        # dyddy = np.vstack((dy, ddy))
        # # print(dyddy.shape)
        # dddy = (jac_ddy @ dyddy).flatten()[0]
        # # Calculate error
        # print(dddy / FX[3] - 1.)

        print('internal error\n', err2)
        iterations += 1
    print(iterations)
    return y2, dy2, ddy
# ======================================================================
def scaled_error[Vars: F64Array[int]](residuals: Vars,
                                      Y: Vars,
                                      rtol: Vars,
                                      atol: Vars) -> float:
    residuals *= residuals
    return (residuals / (rtol * Y * Y + atol)).sum()
# ======================================================================
def step_error2[Vars: F64Array[int],
                Vars2: F64Array[int]](x: float,
               Y_p: Vars2,
               alpha: float,
               a: Vars2,
               step_finish: Callable[[Vars2,
                                      Vars,
                                      Vars2,
                                      F64Array[L[2], L[2]],
                                      Vars2], None],
               f_ddy: Callable[[float, Vars, Vars], Vars],
               rtol: Vars,
               atol: Vars,
               A: Vars2,
               B: F64Array[L[2], L[2]]) -> float:
    n = Y_p.shape[0]//2
    Y = Y_p + alpha * a
    ddy = f_ddy(x, Y[:n], Y[n:])
    Y_step = np.zeros(Y.shape)
    step_finish(Y, ddy, A, B, Y_step)
    residuals = Y - Y_step
    Y_mean = (Y + Y_step) * 0.5
    return scaled_error(residuals, Y_mean, rtol, atol)
# ======================================================================
def plot_alpha(no_show: bool = False):
    print('Plot alpha')
    from matplotlib import pyplot as plt
    from time import perf_counter

    from pade import pade_42
    from poly import poly_33i_prepare, poly_33i_finish, poly_33i, jac_33i
    from references import diffs3, ddy3, jac3

    xp = -0.0275
    x0 = 0.
    x = 0.055
    rtol = 1E-3
    atol = 1E-6
    F0 = diffs3(x0)
    FP = diffs3(xp)
    n = 1

    Dx = x - x0

    Y = np.concatenate(pade_42(F0, FP[:2], xp - x0)(Dx))

    Y_calc = np.zeros(Y.shape)
    A = np.zeros(Y.shape)
    B = np.zeros((2, 2))
    poly_33i_prepare(F0[:3], Dx, A, B)

    #
    ddy = ddy3(x, Y[:n], Y[n:])

    poly_33i_finish(Y, ddy, A, B, Y_calc)

    err = Y - Y_calc
    print(err)
    print('error2', scaled_error(err.copy(), (Y + Y_calc)*0.5, rtol, atol))
    jac_ddy = jac3(x, Y[:n], Y[n:])

    jac_g = jac_33i(Dx, jac_ddy)
    a = -np.linalg.solve(jac_g, err)

    if not no_show:
        alphas = np.linspace(0.5, 1.5)
        error2 = np.zeros(alphas.shape)
        for i, alpha in enumerate(alphas):

            error2[i] = step_error2(x, Y, alpha, a, poly_33i_finish, ddy3,
                                rtol, atol, A, B)

        plt.ion()
        plt.plot(alphas, error2)
        plt.show()
        input()

    Yp = Y
    Y = Yp + a
    a1 = a
    ddy_p = ddy
    ddy = ddy3(x, Y[:n], Y[n:])

    poly_33i_finish(Y, ddy, A, B, Y_calc)

    err_p = err
    err = Y - Y_calc
    print(err)
    print('error2', scaled_error(err.copy(), (Y + Y_calc)*0.5, rtol, atol))
    jac_ddy_p = jac_ddy
    jac_ddy = jac3(x, Y[:n], Y[n:])
    jac_ddy2 = update_jac(jac_ddy_p, a, ddy - ddy_p)
    print('jac prev\n', jac_ddy_p)
    print('jac\n', jac_ddy)
    print('jac2\n', jac_ddy2)

    print('jac change', (jac_ddy2 -jac_ddy)/ (jac_ddy - jac_ddy_p))

    jac_g = jac_33i(Dx, jac_ddy)
    # jac_g = update_jac(jac_g, a, err - err_p)
    t0 = perf_counter()
    a = -np.linalg.solve(jac_g, err)
    print(f'{perf_counter()-t0:.3E} s')

    # angle
    # a2 = a1 + a
    # l1 = np.sqrt(a1.dot(a1))
    # l2 = np.sqrt(a2.dot(a2))
    # angle = np.acos(a2.dot(a1)/(l2 * l1))
    # print('angle', angle / np.pi * 180.)
    # print('length', l2/l1)

    if not no_show:
        for i, alpha in enumerate(alphas):

            error2[i] = step_error2(x, Y, alpha, a, poly_33i_finish, ddy3,
                                rtol, atol, A, B)
        plt.ion()
        plt.plot(alphas, error2)
        plt.show()
        input()

    Yp = Y
    Y = Yp + a
    ddy_p = ddy
    ddy = ddy3(x, Y[:n], Y[n:])

    poly_33i_finish(Y, ddy, A, B, Y_calc)
    err_p = err
    err = Y - Y_calc
    print(err)
    print('error2', scaled_error(err.copy(), (Y + Y_calc)*0.5, rtol, atol))

    jac_ddy = jac3(x, Y[:n], Y[n:])
    # jac_ddy = update_jac(jac_ddy, a, ddy - ddy_p)

    jac_g = jac_33i(Dx, jac_ddy)
    # jac_g = update_jac(jac_g, a, err -err_p)
    a = -np.linalg.solve(jac_g, err)

    Yp = Y
    Y = Yp + a
    a1 = a
    ddy_p = ddy
    ddy = ddy3(x, Y[:n], Y[n:])

    poly_33i_finish(Y, ddy, A, B, Y_calc)

    err = Y - Y_calc
    print(err)
    print('error2', scaled_error(err.copy(), (Y + Y_calc)*0.5, rtol, atol))
# ======================================================================
def update_jac[N_F: int,
               N_X: int,
               Jac: F64Array[N_F, N_X]](jac: Jac,
                                        Delta_x: F64Array[N_X],
                                        Delta_f: F64Array[N_F]) -> Jac:

    return jac + np.outer((Delta_f - jac @ Delta_x)/(Delta_x.dot(Delta_x)),
                          Delta_x)
# ======================================================================

def update_ijac[N_Vars: int,
                Vars: F64Array[N_Vars],
                Jac: F64Array[N_Vars, N_Vars]](ijac: Jac,
                                              Delta_x: Vars,
                                              Delta_f: Vars) -> Jac:
    Dxijax = Delta_x @ ijac
    return ijac + np.outer((Delta_x - ijac @ Delta_f)/(Dxijax.dot(Delta_f)),
                           Dxijax)
# ======================================================================
def error2[N_Y: int, N_ddY: int](error: F64Array[N_Y, 1],
                                 a: F64Array[N_Y, 1],
                                 Y: F64Array[N_Y, 1],
                                 J_P_Y: F64Array[N_Y, N_Y],
                                 J_P_ddy: F64Array[N_Y, N_ddY],
                                 J_e: F64Array[N_ddY, N_Y],
                                 rtol: F64Array[N_Y, 1],
                                 atol: F64Array[N_Y, 1]
           ) -> tuple[float, float, float, Any]:
    tol = rtol * Y * Y + atol

    _1_tol = 1. / tol

    error2 = error.dot(error * _1_tol)

    # Error^2 differential
    dY = -a
    d_error = J_e @ dY
    d_tol = 2. * rtol * Y * dY

    d_error2 = error.dot(2. * d_error - error * d_tol * _1_tol * _1_tol)

    # Error^2 second differential
    d2_error = 0.

    dd_error2 = 2. * (d_error.dot(d_error) + error.dot(d2_error))
    return error2, d_error2, dd_error2, Y
# ======================================================================
def implicit_draft(example: int = 0, no_show: bool = False, step: str = '13i'):
    from references import diffs0, diffs1, diffs2, diffs3, diffs4
    from references import ddy0, ddy1, ddy2, ddy3, ddy4
    from references import jac0, jac1, jac2, jac3, jac4
    from pade import pade_31, pade_32, pade_32, pade_40, pade_41, pade_42
    from poly import poly_13i, poly_33i, poly_33if, poly_43i, poly_44i
    from poly import jac_13i, jac_33i, jac_43i, jac_44i
    from matplotlib import pyplot as plt

    examples = ((diffs0, ddy0, jac0, -0.5, 0.5, 1.5),
            (diffs1, ddy1, jac1, -0.0185, 0., 0.0185),
            (diffs2, ddy2, jac2, -0.03, 0., 0.06),
            (diffs3, ddy3, jac3, -0.0275, 0., 0.055),
            (diffs4, ddy4, jac4, -0.0011, 0., 0.0011),)

    steps = {'13i': (poly_13i, jac_13i),
             '33i': (poly_33i, jac_33i),
             '43i': (poly_43i, jac_43i),
             '44i': (poly_44i, jac_44i)}

    diffs, f_ddy, f_jac_ddy, xp, x0, xi = examples[example]

    Dx = xi - x0

    Fp = diffs(xp)
    F0 = diffs(x0)
    FX = diffs(xi)
    print('original\n', FX)

    _f_step, f_jac_g = steps[step]
    f_step = lambda y, dy, ddy: _f_step(F0, (y, dy, ddy), Dx)

    atol = 1e-14
    rtol = 1e-12
    n = 1

    # Estimate
    estimator = pade_32(F0[:3], Fp[:2], xp - x0)
    # y, dy = pade_42(F0, Fp[:2], xp - x0)(Dx)
    y_e, dy_e = estimator(Dx)
    y_e = np.array((y_e,))
    dy_e = np.array((dy_e,))
    print('estimation error\n', abs(dy_e/FX[0] -1.), abs(dy_e/FX[1] -1.))

    if not no_show:
        plt.ion()

        fig, axs = plt.subplots(3, 1)
        x_plot = np.linspace(x0, xi)
        _y_plot = diffs(x_plot)
        axs[0].plot(x_plot, _y_plot[0], label  = 'original')
        axs[0].plot(xi, y_e, '.', label = f'estimate')
        axs[1].plot(x_plot, _y_plot[1], label  = 'original')
        axs[1].plot(xi, dy_e, '.', label = f'estimate')

        axs[2].set_yscale('log')
        axs[2].axhline()
        axs[0].legend()
        axs[1].legend()
        plt.show()

    # First step with newton iteration
    print('Newton newton')
    y, dy, ddy = newton_newton(Dx, xi, y_e, dy_e, f_step,
                               f_ddy, f_jac_ddy, f_jac_g, atol, rtol, n)
    print('Total error\n', f'{y[0]/FX[0] -1.:.2E}', f'{dy[0] / FX[1] - 1.:.2E}')
    print('Newton secant')
    y, dy, ddy = newton_secant(Dx, xi, y_e, dy_e, f_step,
                               f_ddy, f_jac_ddy, f_jac_g, atol, rtol, n)
    print('Total error\n', f'{y[0]/FX[0] -1.:.2E}', f'{dy[0] / FX[1] - 1.:.2E}')
    print('Fixed secant')
    # y, dy, ddy = fixed_secant(Dx, xi, y_e, dy_e, f_step,
    #                           f_ddy, f_jac_ddy, f_jac_g, atol, rtol, n)
    # print('Total error\n', f'{y[0]/FX[0] -1.:.2E}', f'{dy[0] / FX[1] - 1.:.2E}')

    input()
main = get_main(__name__)
if __name__ == '__main__':
    raise SystemExit(main())
