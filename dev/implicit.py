from collections.abc import Callable

import numpy as np
# ======================================================================
def newton_step(Dx, xi, y, dy, err, f_jac_f: Callable, f_jac_g: Callable):
    n = len(y)
    jac_ddy = f_jac_f(xi, y, dy)
    # print('jac_ddy\n', jac_ddy)
    # Calculate error diffs jacobian
    jac_g = f_jac_g(Dx, jac_ddy)

    # Solve error diffs step
    a = -np.linalg.solve(jac_g, err)
    return a[:n], a[n:]
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

        ay, ady = newton_step(Dx, xi, y2, dy2, err2, f_jac_ddy, f_jac_g)
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
