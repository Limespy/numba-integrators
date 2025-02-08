import numpy as np
# ======================================================================
def poly_13i(F0, FX, Dx):
    return (F0[0] + FX[1] * Dx + 0.5 * Dx * Dx * FX[2]
            ,
            F0[1] + Dx * FX[2])
# ======================================================================
def jac_13i(Dx, Jf):
    n = len(Jf)
    Jg = np.ones((2,2))

    # y = Ay + Dx * dy + 0.5 * Dx * Dx * ddy
    # gy = y - Ay - Dx * dy - 0.5 * Dx * Dx * ddy
    # J(gy) = [I, - Dx] - 0.5 * Dx * Dx * J(ddy)
    Jg[0, 1] = - Dx
    Jg[:n] += 0.5 * Dx * Dx * Jf

    # dy = Ady + Dx * ddy
    # gdy = dy - Ady - Dx * ddy
    # J(gdy) = [0, I] - Dx * J(ddy)
    Jg[1, 0] = 0.
    Jg[n:] += - Dx * Jf
    return Jg
# ======================================================================
def poly_33i(F0, FX, Dx):
    return (F0[0] + F0[1]*Dx/2 + F0[2]*Dx**2/12 + FX[1]*Dx/2 - FX[2]*Dx**2/12
            ,
            Dx*F0[2]/2 + Dx*FX[2]/2 + F0[1]
            # -2*F0[0]/Dx - F0[1] - F0[2]*Dx/6 + 2*FX[0]/Dx + FX[2]*Dx/6
            )
# ======================================================================
def poly_43i(F0, FX, Dx):
    return (F0[0] + 3*Dx*F0[1]/5 + 3*Dx**2*F0[2]/20 + Dx**3*F0[3]/60
            + 2*Dx*FX[1]/5 - Dx**2*FX[2]/20
            ,
            F0[1] + 2*Dx*F0[2]/3 + Dx**2*F0[3]/6 + Dx*FX[2]/3
            # - 5*F0[0]/(2*Dx) - 3*F0[1]/2 - 3*Dx*F0[2]/8 - Dx**2*F0[3]/24
            # + 5*FX[0]/(2*Dx) + Dx*FX[2]/8
            )
# ======================================================================
def jac_43i(Dx, Jf):
    DxJf = Dx * Jf
    n = len(Jf)
    # y = Ay + 2*Dx*dy/5 - Dx**2*ddy/20
    # gy = y - Ay - 2*Dx*dy/5 + Dx**2*ddy/20
    # J(gy) = [I, - 2*Dx*dy/5] + Dx**2/20 * J(ddy)

    # dy = Ady + 5*y/(2*Dx) + Dx*ddy/8
    # gdy = dy - Ady - 5*y/(2*Dx) - Dx*ddy/8
    # J(gdy) = [- 5/(2*Dx), I] - Dx/8 * J(ddy)
    Jg = np.eye(2, dtype = np.float64)
    Jg[0, 1] = - Dx * 2./5.
    Jg[:n, :] += Dx / 20. * DxJf

    Jg[n:, :] += - DxJf / 3.

    # Jgdy = (- 5./(2.*Dx) * y + dy - DxJf / 8.
    # Jg[1, 0] = - 5./(2.*Dx)
    # Jg[n:, :] += - DxJf / 8.

    return Jg
# ======================================================================
def poly_44i(F0, FX, Dx):
    return (F0[0]+ Dx*F0[1]/2 + Dx**2*F0[2]/10 + Dx**3*F0[3]/120
            + Dx**3*FX[3]/120 - Dx**2*FX[2]/10  + Dx*FX[1]/2
            ,
            F0[1] + Dx*F0[2]/2 + Dx**2*F0[3]/12 + Dx*FX[2]/2 - Dx**2*FX[3]/12
            # - Dx**2*F0[3]/60 - Dx**2*FX[3]/60 - Dx*F0[2]/5 + Dx*FX[2]/5 - F0[1] - 2*F0[0]/Dx + 2*FX[0]/Dx
            )
# ======================================================================
def jac_44i(Dx, Jf):
    # DxJf = Dx * Jf
    n = len(Jf)

    # dddy = Jf @ [[dy], [ddy]] = Jf[:, :n] @ dy + Jf[:, n:] @ ddy
    # J(dddy) = Jf[:, :n] + Jf[:, n:] * Jf
    # gy = y - Ay - Dx*dy/2 + Dx**2*ddy/10 - Dx**3*dddy/120
    # J(gy) = [I, 0] - [0, -Dx/2.] + Dx**2/10 * Jf
    #         - Dx**3/120 * (Jf[:, :n] + Jf[:, n:] * Jf)

    Jdddy = (np.hstack((np.eye(n), Jf[:,:n])) + Jf[:, n:] @ Jf)
    Jg = np.eye(n*2, dtype = np.float64)
    Jg[0, 1] = - Dx / 2.
    Jg[:n, :] += Dx**2 / 10. * Jf
    Jg[:n, :] += Dx**3/120 * Jdddy

    # J(gdy) = dy - Dx*ddy/2 + Dx**2*dddy/12
    Jg[n:, :] += - Dx * Jf / 2. + Dx**2/12 * Jdddy
    return Jg
# ======================================================================
