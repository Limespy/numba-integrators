from typing import Literal as L

import numpy as np
from numba_integrators._lnumpy import F64Array
# ======================================================================
def poly_13i(F0, FX, Dx):
    return (F0[0] + FX[1] * Dx + 0.5 * Dx * Dx * FX[2]
            ,
            F0[1] + Dx * FX[2])
# ======================================================================
def jac_13i(Dx, Jddy):
    n = len(Jddy)
    Jg = np.ones((2,2))

    # y = Ay + Dx * dy + 0.5 * Dx * Dx * ddy
    # gy = y - Ay - Dx * dy - 0.5 * Dx * Dx * ddy
    # J(gy) = [I, - Dx] - 0.5 * Dx * Dx * J(ddy)
    Jg[0, 1] = - Dx
    Jg[:n] += 0.5 * Dx * Dx * Jddy

    # dy = Ady + Dx * ddy
    # gdy = dy - Ady - Dx * ddy
    # J(gdy) = [0, I] - Dx * J(ddy)
    Jg[1, 0] = 0.
    Jg[n:] += - Dx * Jddy
    return Jg
# ======================================================================
def poly_33i(F0, FX, Dx):

    return (Dx**2*F0[2]/6 + 2*Dx*F0[1]/3 + Dx*FX[1]/3 + F0[0]
            ,
            -Dx*F0[2]/6 + Dx*FX[2]/6 - F0[1] - 2*F0[0]/Dx + 2*FX[0]/Dx
            )
# ======================================================================
def poly_33i_prepare[Vars: F64Array[int]](F0: tuple[Vars, Vars, Vars, Vars],
                                          Dx: float,
                                          A_out: F64Array[int, int],
                                          B_out: F64Array[L[2],L[2]]) -> None:
    y0, dy0, ddy0 = F0
    n = len(y0)


    A_out[:n] = Dx**2*ddy0/6 + 2*Dx*dy0/3 + y0
    A_out[n:] = -Dx*ddy0/6 - dy0 - 2*y0/Dx

    # B_Y_y_out[:n] = 0.
    B_out[0,0] = Dx / 3.
    # B_out[0,1] = 0.

    # B_Y_y_out[:n] = 0.
    B_out[1,0] = 2. / Dx
    B_out[1,1] = Dx / 6.

# ======================================================================
def poly_33i_finish[Vars: F64Array[int],
                    Vars2: F64Array[int]](Y: Vars2,
                                          ddy: Vars,
                                          A: Vars2,
                                          B: F64Array[L[2], L[2]],
                                          Y_out: Vars2) -> None:
    n = ddy.shape[0]
    Y_out[:] = A
    Y_out[:n] += B[0,0] * Y[n:]
    # Y_out[:n] += B[0,1] * ddy

    Y_out[n:] += B[1, 0] * Y[:n]
    Y_out[n:] += B[1,1] * ddy
# ======================================================================
def jac_33i(Dx, Jddy):
    n_ddy, n_y = Jddy.shape

    Jg = np.eye(n_y)

    # J(gy) = [I, 0] - Dx / 3 [0, I]
    I_ddy = np.eye(n_ddy)
    Jg[:n_ddy, n_ddy:] = - Dx / 3. * I_ddy

    # J(gdy) = [0, I] - 2 / Dx * [I, 0] - Dx/ 6 * J(ddy)
    Jg[n_ddy:, :n_ddy] = -2. / Dx * I_ddy
    Jg[n_ddy:] += - Dx / 6. * Jddy
    return Jg
# ======================================================================
# def poly_33i(F0, FX, Dx):
#     return (F0[0] + F0[1]*Dx/2 + F0[2]*Dx**2/12 + FX[1]*Dx/2 - FX[2]*Dx**2/12
#             ,
#             Dx*F0[2]/2 + Dx*FX[2]/2 + F0[1]
#             )
# # ======================================================================
# def jac_33i(Dx, Jddy):
#     n = len(Jddy)
#     Jg = np.ones((2*n, 2*n))

#     # J(gy) = [I, 0] - Dx / 2 * [0, I] + Dx**2/12*J(ddy)
#     Jg[0:n, n:] = - Dx / 2. * np.eye(n)
#     Jg[:n] += Dx*Dx / 12. * Jddy
#     # J(gdy) = [0, I] - Dx/ 2 * J(ddy)
#     Jg[n:] += - Dx / 2. * Jddy
# # ======================================================================
def poly_33if(F0, FX, Dx):
    return (F0[0] + F0[1]*Dx/2 + F0[2]*Dx**2/12 + FX[1]*Dx/2 - FX[2]*Dx**2/12
            ,
            -2*F0[0]/Dx - F0[1] - F0[2]*Dx/6 + 2*FX[0]/Dx + FX[2]*Dx/6
            )
# ======================================================================
def poly_43i(F0, FX, Dx):
    """Skips:
    0: 0, 2,
    1: 1"""
    return (F0[0] + 3*Dx*F0[1]/4 + Dx**2*F0[2]/4 + Dx**3*F0[3]/24 + Dx*FX[1]/4
            ,
             - 5*F0[0]/(2*Dx) - 3*F0[1]/2 - 3*Dx*F0[2]/8 - Dx**2*F0[3]/24
              + 5*FX[0]/(2*Dx) + Dx*FX[2]/8
            )
# ======================================================================
def jac_43i(Dx, Jddy):
    # DxJf = Dx * Jddy
    n = len(Jddy)

    Jg = np.eye(2, dtype = np.float64)

    # J(gy) = y - Dx/4*[0, 1]
    Jg[0:n, n:] = - Dx / 4.  * np.eye(n)
    # Jg[:n, :] += Dx / 20. * DxJf
    # J(gdy) = dy - 5/(2*Dx) * [1,0] + Dx*Jddy/8
    Jg[n:, 0:n] = - 5/(2*Dx)  * np.eye(n)
    Jg[n:, :] += - Dx * Jddy / 8.
    return Jg
# ======================================================================
# def poly_43i(F0, FX, Dx):
#     return (F0[0] + 3*Dx*F0[1]/5 + 3*Dx**2*F0[2]/20 + Dx**3*F0[3]/60
#             + 2*Dx*FX[1]/5 - Dx**2*FX[2]/20
#             ,
#             F0[1] + 2*Dx*F0[2]/3 + Dx**2*F0[3]/6 + Dx*FX[2]/3)
# # ======================================================================
# def jac_43i(Dx, Jddy):
#     DxJf = Dx * Jddy
#     n = len(Jddy)
#     # y = Ay + 2*Dx*dy/5 - Dx**2*ddy/20
#     # gy = y - Ay - 2*Dx*dy/5 + Dx**2*ddy/20
#     # J(gy) = [I, - 2*Dx*dy/5] + Dx**2/20 * J(ddy)

#     # dy = Ady + 5*y/(2*Dx) + Dx*ddy/8
#     # gdy = dy - Ady - 5*y/(2*Dx) - Dx*ddy/8
#     # J(gdy) = [- 5/(2*Dx), I] - Dx/8 * J(ddy)
#     Jg = np.eye(2, dtype = np.float64)
#     Jg[0, 1] = - Dx * 2./5.
#     Jg[:n, :] += Dx / 20. * DxJf

#     Jg[n:, :] += - DxJf / 3.

#     # Jgdy = (- 5./(2.*Dx) * y + dy - DxJf / 8.
#     # Jg[1, 0] = - 5./(2.*Dx)
#     # Jg[n:, :] += - DxJf / 8.

#     return Jg
# ======================================================================
# def poly_43i(F0, FX, Dx):
#     return (F0[0] + 3*Dx*F0[1]/5 + 3*Dx**2*F0[2]/20 + Dx**3*F0[3]/60
#             + 2*Dx*FX[1]/5 - Dx**2*FX[2]/20
#             ,
#             - 5*F0[0]/(2*Dx) - 3*F0[1]/2 - 3*Dx*F0[2]/8 - Dx**2*F0[3]/24
#             + 5*FX[0]/(2*Dx) + Dx*FX[2]/8)
# # ======================================================================
# def jac_43i(Dx, Jddy):
#     DxJf = Dx * Jddy
#     n = len(Jddy)

#     Jg = np.eye(2, dtype = np.float64)
#     Jg[0, 1] = - Dx * 2./5.
#     Jg[:n, :] += Dx / 20. * DxJf

#     # Jg[n:, :] += - DxJf / 3.

#     # Jgdy = (- 5./(2.*Dx) * y + dy - DxJf / 8.
#     Jg[1, 0] = - 5./(2.*Dx)
#     Jg[n:, :] += - DxJf / 8.

#     return Jg
# ======================================================================
# def poly_44i(F0, FX, Dx):
#     return (Dx**3*F0[3]/120 + Dx**3*FX[3]/120 + Dx**2*F0[2]/10 - Dx**2*FX[2]/10 + Dx*F0[1]/2 + Dx*FX[1]/2 + F0[0]
#             ,
#             -Dx**2*F0[3]/60 - Dx**2*FX[3]/60 - Dx*F0[2]/5 + Dx*FX[2]/5 - F0[1] - 2*F0[0]/Dx + 2*FX[0]/Dx
#             )
# ======================================================================
def poly_44i(F0, FX, Dx):
    """skips:
        0: 0
        1: 1"""
    return (F0[0]+ Dx*F0[1]/2 + Dx**2*F0[2]/10 + Dx**3*F0[3]/120
            + Dx**3*FX[3]/120 - Dx**2*FX[2]/10  + Dx*FX[1]/2
            ,
            -Dx**2*F0[3]/60 - Dx**2*FX[3]/60 - Dx*F0[2]/5 + Dx*FX[2]/5 - F0[1] - 2*F0[0]/Dx + 2*FX[0]/Dx
            )
# ======================================================================
# def poly_44i(F0, FX, Dx):
#     """skips:
#         0: 0
#         1: 1"""
#     return (F0[0]+ Dx*F0[1]/2 + Dx**2*F0[2]/10 + Dx**3*F0[3]/120
#             + Dx**3*FX[3]/120 - Dx**2*FX[2]/10  + Dx*FX[1]/2
#             ,
#             F0[1] + Dx*F0[2]/2 + Dx**2*F0[3]/12 + Dx*FX[2]/2 - Dx**2*FX[3]/12
#             # - Dx**2*F0[3]/60 - Dx**2*FX[3]/60 - Dx*F0[2]/5 + Dx*FX[2]/5 - F0[1] - 2*F0[0]/Dx + 2*FX[0]/Dx
#             )
# ======================================================================
def jac_44i(Dx, Jddy):
    # DxJf = Dx * Jddy
    n = len(Jddy)

    # dddy = Jddy @ [[dy], [ddy]] = Jddy[:, :n] @ dy + Jddy[:, n:] @ ddy
    # J(dddy) = Jddy[:, :n] + Jddy[:, n:] * Jddy
    # gy = y - Ay - Dx*dy/2 + Dx**2*ddy/10 - Dx**3*dddy/120
    # J(gy) = [I, 0] - [0, -Dx/2.] + Dx**2/10 * Jddy
    #         - Dx**3/120 * (Jddy[:, :n] + Jddy[:, n:] * Jddy)

    Jdddy = (np.hstack((np.eye(n), Jddy[:,:n])) + Jddy[:, n:] @ Jddy)
    Jg = np.eye(n*2, dtype = np.float64)
    Jg[0, 1] = - Dx / 2.
    Jg[:n, :] += Dx**2 / 10. * Jddy
    Jg[:n, :] += Dx**3/120 * Jdddy

    # J(gdy) = dy - Dx*ddy/2 + Dx**2*dddy/12
    Jg[n:, :] += - Dx * Jddy / 2. + Dx**2/12 * Jdddy
    return Jg
# ======================================================================
