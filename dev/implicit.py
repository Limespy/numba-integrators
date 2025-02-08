def implicit_draft(example: int = 0, no_show: bool = False, step: str = '13i'):
    import numpy as np
    from references import diffs0, diffs1, diffs2, diffs3, diffs4
    from references import ddy0, ddy1, ddy3, ddy4
    from references import jac0, jac1, jac2, jac3, jac4
    from pade import pade_31, pade_32, pade_32, pade_40, pade_41, pade_42
    from poly import poly_13i, poly_33i, poly_43i, poly_44i
    from poly import jac_13i, jac_43i, jac_44i
    from matplotlib import pyplot as plt

    examples = ((diffs0, ddy0, jac0, -0.5, 0.5, 1.5),
            (diffs1, ddy1, jac1, -0.0185, 0., 0.0185),
            (diffs2, jac2, -0.03, 0., 0.06),
            (diffs3, ddy3, jac3, -0.0275, 0., 0.055),
            (diffs4, ddy4, jac4, -0.00055, 0., 0.0011),)

    steps = {'13i': (poly_13i, jac_13i),
             '33i': (poly_33i, ),
             '43i': (poly_43i, jac_43i),
             '44i': (poly_44i, jac_44i)}

    diffs, f_ddy, jac_f, xp, x0, xi = examples[example]

    Dx = xi - x0

    Fp = diffs(xp)
    F0 = diffs(x0)
    FX = diffs(xi)
    print('original\n', FX)

    f_step, f_jac_g = steps[step]

    # Estimate

    estimator = pade_42(F0, Fp[:2], xp - x0)

    # y, dy = pade_42(F0, Fp[:2], xp - x0)(Dx)
    y, dy = pade_42(F0, Fp[:2], xp - x0)(Dx)
    print('estimation error\n', abs(y/FX[0] -1.), abs(dy/FX[1] -1.))

    ydy = np.array((y, dy))
    # print(ydy)
    plt.ion()
    plt.show()

    if not no_show:
        fig, axs = plt.subplots(3, 1)
        x_plot = np.linspace(x0, xi)
        _y_plot = diffs(x_plot)
        axs[0].plot(x_plot, _y_plot[0], label  = 'original')
        axs[0].plot(xi, y, '.', label = f'estimate')
        axs[1].plot(x_plot, _y_plot[1], label  = 'original')
        axs[1].plot(xi, dy, '.', label = f'estimate')

        axs[2].set_yscale('log')
        axs[2].axhline()
        axs[0].legend()
        axs[1].legend()
    # input()
    # Loop
    for iteration in range(5):

        y, dy = ydy
        # calculate ddy

        ddy = f_ddy(Dx, y, dy)

        # Calculate ddy jacobian

        jac_ddy = jac_f(xi, y, dy)

        # print('jac_ddy\n', jac_ddy)
        # Calculate error diffs jacobian
        jac_g = f_jac_g(Dx, jac_ddy)

        dyddy = np.vstack((dy, ddy))
        # print(dyddy.shape)
        dddy = (jac_ddy @ dyddy).flatten()[0]
        # Calculate error
        print(dddy / FX[3] - 1.)

        y_calc, dy_calc = f_step(F0, (y, dy, ddy, dddy), Dx)
        # print('y, dy calc', y_calc, dy_calc)
        err = np.hstack((y - y_calc, dy - dy_calc))
        if not no_show:
            axs[2].plot(iteration, abs(err[0]), '.', color = 'black')
        print('internal error\n', err)
        print('full error\n', (y_calc / FX[0] - 1), (dy_calc / FX[1] - 1))

        # Solve error diffs step

        a = np.linalg.solve(jac_g, err)
        # Take step

        ydy -= a
        if not no_show:
            axs[0].plot(xi, ydy[0], '.', label = f'{iteration}')
            axs[0].legend()
            axs[1].plot(xi, ydy[1], '.', label = f'{iteration}')
            axs[1].legend()
            plt.pause(0.1)

    # total error estimation
    y_e, dy_e = poly_44i(F0, (y, dy, ddy, dddy), Dx)
    print('error estimate\n', abs(y_e / y - 1), abs(dy_e / dy - 1))
    input()
