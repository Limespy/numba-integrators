import numpy as np
from implicit import implicit_draft
from limedev.CLI import get_main
from pade import pade_31
from pade import pade_32
from pade import pade_32i
from pade import pade_40
from pade import pade_41
from pade import pade_42
from pade import pade_42i
from poly import poly_33i
from poly import poly_33if
from poly import poly_43i
from poly import poly_44i
# ======================================================================
# @nb.njit
def frange(n: int, dtype: type = np.float64):
    array = np.ones((n,), dtype)
    m = dtype(2)
    _1 = dtype(1)
    for i in range(2, n):
        array[i] = array[i - 1] * m
        m += _1
    return array
# ======================================================================
def erange(n, x):
    out = np.ones(n)
    for i in range(1, n):
        out[i] = out[i - 1] * x
    return out

# ======================================================================
def _construct_NM(n_diffs, length, dtype):
    NM = np.zeros((n_diffs, length), dtype = dtype)
    NM[0] = 1.
    start_col = 1
    for i_row in range(1, n_diffs):
        NM[i_row, start_col] = NM[i_row - 1, start_col]
        m = 2.
        for i_col in range(start_col + 1, length):
            NM[i_row, i_col] = NM[i_row - 1, i_col] * m
            m += 1.
        start_col += 1
    return NM[:,:n_diffs], NM[:,n_diffs:]
# ======================================================================
def _implicit_parameters_construct(n_diffs: int, Dx: float,
                                   dtype: type = np.float64):
    length = 2 * n_diffs
    F = frange(n_diffs)
    return F, *_construct_NM(n_diffs, length, dtype), erange(length, Dx)
# ======================================================================
def matrices():
    import sympy as sp
    x = sp.Symbol('x', real = True)
    X = sp.Matrix(3, 3, lambda i,j: x**i if i == j else 0)
    IX = sp.Matrix(3, 3, lambda i,j: x**(-i) if i == j else 0)

    A = sp.Matrix(3, 3, lambda i,j: sp.Symbol(f'a{i}{j}', real = True))
    f = sp.Matrix(3, 1, lambda i,j: sp.Symbol(f'f{i}', real = True))
    print(X)
    print(IX)
    print(A)
    print(IX * A * X)
    print(IX * A * X * f)
# ======================================================================
def _pade_R(d0: int, dx: int = 0, skip: int = -1):
    import sympy as sp
    from math import ceil

    total = d0 + dx
    if skip >= 0:
        total -= 1
    m = total // 2 - 1
    n = ceil(total / 2.)
    x = sp.Symbol('x', real = True)

    A = tuple(sp.Symbol(f'a{j}', real = True) for j in range(m+1))
    B = tuple(sp.Symbol(f'b{k}', real = True) for k in range(1, n+1))

    R = (sum(A[j] * x ** j for j in range(m+1))
         / (1 + sum(B[k-1] * x ** k for k in range(1, n+1))))
    Rs = [R]
    for i in range(1, max(d0, dx)):
        Rs.append(sp.diff(Rs[i-1], x))

    equations = []
    F0 = tuple(sp.Symbol(f'f0{i}', real = True) for i in range(d0))
    for r, f in zip(Rs, F0):
        equations.append(sp.Equality(f, r.subs(x, 0)))
    FX = tuple(sp.Symbol(f'fx{i}', real = True) for i in range(dx))
    for i, (r, f) in enumerate(zip(Rs, FX)):
        if i != skip:
            equations.append(sp.Equality(f, r))
    return x, A, B, F0, FX, equations
# ======================================================================
def jacobian_update():
    import sympy as sp
    J = sp.Matrix(2,4, lambda i,j: sp.Symbol(f'j{i}{j}'))
    Delta_y = sp.Matrix(4,1, lambda i,j: sp.Symbol(f'Dy{i}'))
    Delta_f = sp.Matrix(2,1, lambda i,j: sp.Symbol(f'Df{i}'))
    Delta_J = (Delta_f - J * Delta_y) * Delta_y.T
    print(Delta_J[0,0])
    print(Delta_J[0,1])
    print(Delta_J[1,0])
    print(Delta_J[1,1])
# ======================================================================
def error_differentials():
    import sympy as sp
    alpha = sp.Symbol('alpha', real = True)
    a = sp.MatrixSymbol('a', 2, 1)
    Yp = sp.MatrixSymbol('Yp', 2, 1)
    Y = Yp - alpha * a
    R = sp.MatrixSymbol('R', 2, 1)
    A = sp.MatrixSymbol('A', 2, 1)
    T = sp.hadamard_product(R, Y, Y) + A
    ddy = sp.MatrixSymbol('P_ydy', 1, 1)
    P_ydy = sp.MatrixSymbol('P_ydy', 2, 2)
    P_ddy = sp.MatrixSymbol('P_ddy', 2, 1)
    f_step = P_ydy * Y + P_ddy * ddy
    # print(T)
    # print(T.diff(Y))

    I = sp.MatrixSymbol('I', 2, 2)
    J = sp.MatrixSymbol('J', 2, 2)
    E = I * Yp / (J * Yp)
    print(E)
    sp.simplify(E)
# ======================================================================
def _print_pade(solutions, params):
    for solution in solutions:
        print('\nx2 = x*x\n'
              'x3 = x2*x\n'
              'x4 = x3*x\n'
              'x5 = x4*x\n'
              'f012 = f01 * f01\n'
              'f013 = f012 * f01\n'
              'f014 = f013 * f01\n'
              'f015 = f014 * f01\n'
              'f022 = f02 * f02\n'
              'f023 = f022 * f02\n'
              'f024 = f023 * f02\n'
              'f025 = f024 * f02\n'
              'f032 = f03 * f03\n'
              'f033 = f032 * f03\n'
              'f034 = f033 * f03\n'
              'f035 = f034 * f03\n'
              'fx02 = fx0 * fx0\n'
              'fx03 = fx02 * fx0\n'
              'fx04 = fx03 * fx0\n'
              'fx05 = fx04 * fx0\n'
              'fx12 = fx11 * fx1\n'
              'fx13 = fx12 * fx1\n'
              'fx14 = fx13 * fx1\n'
              'fx15 = fx14 * fx1\n'
              'a0 = f00\n'
              'a02 = a0*a0\n'
              'a03 = a02*a0\n'
              'a04 = a03*a0\n'
              'a05 = a04*a0\n'
              )
        for p, e in zip(params, solution):
            str_e = str(e
                ).replace('**2', '2'
                ).replace('**3', '3'
                ).replace('**4', '4'
                ).replace('**5', '5'
                # ).replace('f01**2', 'f012'
                # ).replace('f02**2', 'f022'
                # ).replace('f02**3', 'f023'
                # ).replace('fx0**2', 'fx02'
                # ).replace('fx1**2', 'fx12'
                # ).replace('a0**2', 'a02'
                # ).replace('a0**3', 'a03'
                # ).replace('a0**2', 'a02'
                )
            print(f'{p} = {str_e}')
# ======================================================================
def calc_pade(d0: int = 2, d1: int = 2):
    import sympy as sp
    from sympy.solvers.solveset import nonlinsolve

    x, A, B, F0, FX, equations = _pade_R(d0, d1)
    for eq in equations:
        print(eq)
    params = (*A[1:], *B)
    solutions = nonlinsolve(equations[1:], *params)
    _print_pade(solutions, params)
# ======================================================================
def diff_pade(m: int, n: int):
    import sympy as sp

    x = sp.Symbol('Dx', real = True)
    A = tuple(sp.Symbol(f'a{j}', real = True) for j in range(m+1))
    B = tuple(sp.Symbol(f'b{k}', real = True) for k in range(1, n+1))
    R = (sum(A[j] * x ** j for j in range(m+1))
         / (1 + sum(B[k-1] * x ** k for k in range(1, n+1))))
    print(R)
    print(sp.diff(R, x))
# ======================================================================
def calc_pade_partial(d0: int = 2, d1: int = 2, skip: int = 0):
    import sympy as sp
    from sympy.solvers.solveset import nonlinsolve

    x, A, B, F0, FX, equations = _pade_R(d0, d1, skip = skip)
    for eq in equations:
        print(eq)
    params = (*A[1:], *B)
    solutions = nonlinsolve(equations[1:], *params)
    _print_pade(solutions, params)
# ======================================================================
def eval_pade(x, a, b):
    num = a[-1] * x
    for _a in reversed(a[1:-1]):
        num += _a
        num *= x
    num += a[0]

    den = b[-1] * x
    for _b in reversed(b[:-1]):
        den += _b
        den *= x
    den += 1.
    return num / den
# ======================================================================
def jacobian():
    import sympy as sp
    x = sp.Symbol('x', real = True)
    y = sp.Matrix(2, 1, lambda i,j: sp.Symbol(f'y{i}{j}', real = True))
    # A = sp.Matrix(2, 2, lambda i,j: sp.Symbol(f'a{i}{j}', real = True))
    dy = sp.Matrix(((-y[0, 0]*y[1,0]), (y[0, 0]/y[1,0])))
    # dy = sp.Matrix(2, 1, lambda i,j: sp.Symbol(f'dy{i}{j}', real = True))
    J_dy = dy.jacobian(y)
    print('J_dy\n', J_dy)
    ddy = J_dy * dy
    print('ddy\n', ddy)
    Dy = dy * x + x**2 / 2 * ddy
    print('Delta y\n', Dy)
    J_ddy = ddy.jacobian(y)
    print('J_ddy\n', J_ddy)
    # print(J_dy*J_dy)
# ======================================================================
def eval_poly(x, coeffs):
    out = x * coeffs[-1]
    for c in reversed(coeffs[1:-1]):
        out += c
        out *= x
    out += coeffs[0]
    return out
# ======================================================================
def calc_poly_i(d0: int = 3, dx: int = 3,
                target: int = 0,
                skip: tuple[int, ...] = ()):
    import sympy as sp

    _1 = sp.sympify(1)
    d_max = max(d0, dx)
    x = sp.Symbol('Dx', real = True)
    N_np, M_np = _construct_NM(d0, d0 + dx - len(skip), np.int32)
    _M = sp.Matrix(dx, dx - len(skip), lambda i,j: M_np[i, j])
    _N = sp.Matrix(dx, d0, lambda i,j: N_np[i, j])
    X = sp.Matrix(d_max, d_max, lambda i,j: x**i if i == j else 0)
    F0 = sp.Matrix(d0, 1, lambda i,j: sp.Symbol(f'F0[{i}]', real = True))
    FX = sp.Matrix(dx, 1, lambda i,j: sp.Symbol(f'FX[{i}]', real = True))
    # _X = sp.Matrix(d - 1, d - 1, lambda i,j: (sp.Symbol(f'_x{i}', real = True)
    #                                   if i == j else 0))
    F_np = frange(d0, np.int32)
    IF = sp.Matrix(d0, d0, lambda i,j: _1 / F_np[i] if i == j else 0)

    mv = _M[target, :]
    nv = _N[target, :]
    ixv = 1/x**target
    _X = X[:dx, :dx].copy()
    for s in sorted(skip, reverse = True):
        FX.row_del(s)
        _X.row_del(s)
        _X.col_del(s)
        _N.row_del(s)
        _M.row_del(s)
    IM = _M.inv()
    # O = (nv - mv * IM * _N) * IF

    # V = IX[:d] * (N - M[:, :-1] @ O) @ (X[:d] * PA)
    # V = O * (X[:d] * F0)
    # K = mv * IM
    # DB_new = d_ixv @ (V + K @ d_X @ _DB)
    # DB_new = d_ixv @ V + d_ixv @ K @ d_X @ _DB
    A = (ixv * (nv - mv * IM * _N) * IF * X[:d0,:d0] * F0)
    B = ixv * mv * IM * _X
    print('A\n',A)
    print('B\n',B)
    DB_new = A + B @ FX
    print(DB_new[0,0])
# ======================================================================
def diff_poly(coeffs):
    return coeffs[1:] * np.arange(1., len(coeffs), dtype = np.float64)
# ======================================================================
def implicit_parameters(example: int = 0, no_show: bool = False):
    from matplotlib import pyplot as plt
    from references import diffs0, diffs1, diffs2, diffs3, diffs4
    inv = np.linalg.inv
    info: tuple[list[str], list[str], list[str]] = ([], [], [])

    examples = ((diffs0, -0.5, 0.5, 1.5),
                (diffs1, -0.0185, 0., 0.0185),
                (diffs2, -0.03, 0., 0.06),
                (diffs3, -0.0275, 0., 0.055),
                (diffs4, -0.00055, 0., 0.0011),)

    f, xp, xa, xb = examples[example]

    Dxp = xp - xa
    Dx = xb - xa
    Dp = f(xp)
    F0 = f(xa)
    FX = f(xb)
    d = len(F0)
    def err(v, ref):
        return f'{np.log10(abs(v/ref - 1)):.1f}'

    # print('FX\n', FX)
    F, N, M, X = _implicit_parameters_construct(d, Dx)

    # f_pade_22 = pade_22(F0[:2], Dp[:2], Dxp)
    # f_pade_23 = pade_23(Dp[:2], F0[:3], Dx)
    f_pade_31 = pade_31(F0[:3], Dp[:1], Dxp)
    f_pade_32 = pade_32(F0[:3], Dp[:2], Dxp)
    f_pade_32i = pade_32i(F0[:3], FX[1:2], Dx)
    f_pade_40 = pade_40(F0[:4], Dp[:0], Dxp)
    f_pade_41 = pade_41(F0[:4], Dp[:1], Dxp)
    f_pade_42 = pade_42(F0[:4], Dp[:2], Dxp)
    f_pade_42i = pade_42i(F0[:4], FX[1:2], Dx)
    coeffs_40 = F0 / F
    coeffs_30 = coeffs_40[:3]
    coeffs_340 = coeffs_40.copy()
    coeffs_340[3] *= 0.5
    f_poly_20 = lambda x: eval_poly(x, coeffs_40[:2])
    f_poly_30 = lambda x: eval_poly(x, coeffs_30)
    f_poly_340 = lambda x: eval_poly(x, coeffs_340)
    f_poly_40 = lambda x: eval_poly(x, coeffs_40)
    # info[2].append('Simple')
    # info[0].append(err(F0[0] + F0[1]* Dx,
    #                    FX[0]))
    # info[1].append(err(F0[1] + F0[2] * Dx,
    #                    FX[1]))

    # info[2].append('Forward')
    # info[0].append(err(eval_poly(Dx, F0 / F), FX[0]))
    # info[1].append(err(eval_poly(Dx, F0[1:] / F[:-1]), FX[1]))


    # info[2].append('Backward')
    # info[0].append(err(F0[0] + FX[1]* Dx + FX[2]/2*Dx*Dx + FX[3]/6*Dx**3,
    #                    FX[0]))
    # info[1].append(err(F0[1] + FX[2]*Dx + FX[3]/2*Dx*Dx,
    #                    FX[1]))

    # info[2].append('Pade22')
    # info[0].append(err(f_pade_22(Dx), FX[0]))
    # info[1].append(err(0., FX[1]))

    info[2].append('Poly30')
    info[0].append(err(f_poly_30(Dx), FX[0]))
    info[1].append(err(0., FX[1]))

    info[2].append('Poly40')
    info[0].append(err(f_poly_40(Dx), FX[0]))
    info[1].append(err(0., FX[1]))

    info[2].append('Pade31')
    _pade31 = f_pade_31(Dx)
    info[0].append(err(_pade31[0], FX[0]))
    info[1].append(err(_pade31[1], FX[1]))

    info[2].append('Pade32')
    _pade32 = f_pade_32(Dx)
    info[0].append(err(_pade32[0], FX[0]))
    info[1].append(err(_pade32[1], FX[1]))

    info[2].append('Pade40')
    _pade40 = f_pade_40(Dx)
    info[0].append(err(_pade40[0], FX[0]))
    info[1].append(err(_pade40[1], FX[1]))

    info[2].append('Pade41')
    info[0].append(err(f_pade_41(Dx), FX[0]))
    info[1].append(err(0., FX[1]))

    info[2].append('Pade42')
    _pade42 = f_pade_42(Dx)
    info[0].append(err(_pade42[0], FX[0]))
    info[1].append(err(_pade42[1], FX[1]))

    info[2].append('Pade42i')
    _pade42i = f_pade_42i(Dx)
    info[0].append(err(_pade42i[0], FX[0]))
    info[1].append(err(_pade42i[1], FX[1]))

    info[2].append('Poly33i')
    _p33i = poly_33i(F0, FX, Dx)
    info[0].append(err(_p33i[0], FX[0]))
    info[1].append(err(_p33i[1], FX[1]))

    info[2].append('Poly33if')
    _poly_33if = poly_33if(F0, FX, Dx)
    info[0].append(err(_poly_33if[0], FX[0]))
    info[1].append(err(_poly_33if[1], FX[1]))

    info[2].append('Poly43i')
    _p43i = poly_43i(F0, FX, Dx)
    info[0].append(err(_p43i[0], FX[0]))
    info[1].append(err(_p43i[1], FX[1]))

    info[2].append('Poly44i')
    _p44i = poly_44i(F0, FX, Dx)
    info[0].append(err(_p44i[0], FX[0]))
    info[1].append(err(_p44i[1], FX[1]))

    # print(poly_43i(F0, FX, Dx) - FX[0])
    # print(poly_44i(F0, FX, Dx) - FX[0])

    # print('F\n', F)
    # print('NM\n', NM)
    # print('X\n', X)
    F0 = F0.reshape(-1,1)
    FX = FX.reshape(-1,1)
    # info[2].append('Poly44i')
    IF = 1./F.reshape(-1,1)
    X = X.reshape(-1,1)
    IX = 1./X
    coeffs = []
    for v in (0, 1):
        mv = M[:, :-1]
        nv = N
        d_ixv = np.diag(IX[:d].flatten())
        # Inverting the rest of the M
        _N = np.concatenate((N[:v, :], N[v+1:-1, :]))
        _M = np.concatenate((M[:v, :-2], M[v+1:-1, :-2]))
        _X = np.vstack((X[:v], X[v+1:-1]))
        # _IX = np.vstack((IX[:v], IX[v+1:-1]))
        _DB = np.vstack((FX[:v,:], FX[v+1:-1, :]))
        # print('_DB\n',_DB)

        # print('IX\n', IX)
        # _IM = inv(_M)
        # print('_IM\n', _IM)

        PA = IF * F0
        # PB = IX[d:] * (inv(M) @ (X[:d] * FX - N @ (X[:d] * PA)))
        PB = IX[d:-2] * (inv(_M) @ (_X[:d-2] * _DB - _N @ (X[:d] * PA)))

        # DB_new = IX[:d] * (N @ (X[:d] * PA) + M[:, :-1] @ (X[d:-1] * PB))
        # DB_new = (IX[:d] * N @ (X[:d] * PA)
        #           + IX[:d] * M[:, :-1] @ (X[d:-1] * PB))

        # H = IX[:d] * N
        # J =
        # XPA = X[:d] * PA
        # DB_new = (H @ (XPA) + J @ inv(_M) @ (_X[:d-1] * _DB - _N @ XPA))
        # IM = inv(_M)
        # # K = (ixv * mv) @ IM
        # # L = K @ _N
        # # DB_new = (H @ (XPA) + K @ (_X[:d-1] * _DB) - L @ XPA)
        # # V = (H - L)
        # dIF = np.diag(IF.flatten())
        # O = (nv - mv @ IM @ _N) @ dIF

        # # V = IX[:d] * (N - M[:, :-1] @ O) @ (X[:d] * PA)
        # V = O @ (X[:d] * F0)
        # K = mv @ IM
        # d_X = np.diag(_X[:d-1].flatten())
        # # DB_new = d_ixv @ (V + K @ d_X @ _DB)
        # # DB_new = d_ixv @ V + d_ixv @ K @ d_X @ _DB
        # A = d_ixv @ V
        # B = d_ixv @ K @ d_X
        # DB_new = A + B @ _DB

        coeffs.append(np.vstack((PA, PB)).flatten())

    x_plot = np.linspace(xp, xb)
    x_plot_i = np.linspace(xa, xb)
    Dx_plot = np.linspace(Dxp, Dx)
    Dx_plot_i = np.linspace(0., Dx)

    lengths = tuple(len(h) for h in info[2])
    print(*info[2])
    print(*(f'{v:^{l}}' for v, l in zip(info[0], lengths)))
    print(*(f'{v:^{l}}' for v, l in zip(info[1], lengths)))

    if not no_show:
        y_plot = f(x_plot)
        # print(eval_poly(0., coeffs[0]))
        plt.plot(x_plot, y_plot[0], label = 'original')
        plt.plot(x_plot_i, eval_poly(Dx_plot_i, coeffs[0]), label = 'poly43i')
        # _poly_43i = poly_43i(F0, FX, Dx_plot_i)
        # plt.plot(x_plot_i, _poly_43i[0], label = 'poly 43i')
        # _poly_44i = poly_44i(F0, FX, Dx_plot_i)
        # plt.plot(x_plot_i, _poly_44i[0], label = 'poly 44i')
        plt.plot(x_plot_i, f_poly_20(Dx_plot_i), label = 'poly 20')
        plt.plot(x_plot_i, f_poly_30(Dx_plot_i), label = 'poly 30')
        plt.plot(x_plot_i, f_poly_40(Dx_plot_i), label = 'poly 40')
        plt.plot(x_plot_i, f_poly_340(Dx_plot_i), label = 'poly 340')

        # y_backward = F0[0] + FX[1]* Dx_plot + FX[2]/2*Dx_plot**2 + FX[3]/6*Dx_plot**3
        # plt.plot(x_plot, y_backward, label = 'backward')



        # plt.plot(x_plot, f_pade_22(Dx_plot), label = 'pade 22')
        # plt.plot(x_plot, f_pade_31(Dx_plot), label = 'pade 31')
        # plt.plot(x_plot, f_pade_32(Dx_plot), label = 'pade 32')
        # plt.plot(x_plot, f_pade_23(Dx_plot+Dx), label = 'pade 23')

        plt.plot(x_plot_i, f_pade_40(Dx_plot_i), label = 'pade 40')
        # plt.plot(x_plot, f_pade_41(Dx_plot), label = 'pade 41')
        plt.plot(x_plot, f_pade_42(Dx_plot)[0], label = 'pade 42')
        # plt.plot(x_plot, f_pade_32i(Dx_plot), label = 'pade 32i')
        plt.plot(x_plot_i, f_pade_42i(Dx_plot_i), label = 'pade 42i')

        # plt.ylim(0., 1.)
        plt.legend()
        plt.show()
# ======================================================================
def minimiser():
    from matplotlib import pyplot as plt

    def function2min(x):
        return x + np.exp(x) + np.exp(-x)
    def D_function2min(x):
        return 1. + np.exp(x) - np.exp(-x)

    plt.ion()
    x_plot = np.linspace(-2., 2.)
    y_plot = function2min(x_plot)
    plt.plot(x_plot, y_plot, label = 'function')

    x_1 = 2.
    e2_1 = function2min(x_1)
    x_2 = x_1 - e2_1 / D_function2min(x_1)
    e2_2 = function2min(x_2)

    plt.plot(x_1, e2_1, '.', label = '0')
    plt.plot((x_2, x_1), (0., e2_1))
    plt.plot(x_2, e2_2, '.', label = '1')
    plt.grid()
    plt.legend()
    input()
    for i in range(2, 8):
        De2_2 = D_function2min(x_2)
        print(e2_1 / e2_2 - 1.)
        Rp = (e2_2 - e2_1) / De2_2
        # Debugging
        # p0 = e2_2
        # p1 = De2_2
        # p2 = (e2_1 - p0 - p1 * (x_1 - x_2))/(x_1 - x_2)**2
        # Dx = x_plot - x_2
        # plt.plot(x_plot, p0 + p1 * Dx + p2 * Dx*Dx)


        x_1, x_2 = x_2, ((x_2*x_2 - x_1 * x_1) * 0.5 - Rp * x_2)/(x_2 - x_1 - Rp)
        # print(e2_1 / e2_2 - 1.)
        e2_1 = e2_2
        e2_2 = function2min(x_2)
        plt.plot(x_2, e2_2, '.', label = str(i))
        plt.legend()
        input()

    plt.legend()
    plt.show()

    input()
# ======================================================================
main = get_main(__name__)
