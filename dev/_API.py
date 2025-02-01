from limedev.CLI import get_main
import numpy as np
import numba as nb
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
def _implicit_parameters_construct(n_diffs: int, Dx):
    length = 2 * n_diffs - 1
    F = frange(n_diffs)
    NM = np.zeros((n_diffs, length), dtype = np.float64)
    NM[0] = 1.
    start_col = 1
    for i_row in range(1, n_diffs):
        NM[i_row, start_col] = NM[i_row - 1, start_col]
        m = 2.
        for i_col in range(start_col + 1, length):
            NM[i_row, i_col] = NM[i_row - 1, i_col] * m
            m += 1.
        start_col += 1
    N = NM[:,:n_diffs]
    M = NM[:,n_diffs:]
    # X
    
    # X
    X = np.ones(NM.shape)
    for i_col in range(1, length):
        X[0, i_col] = X[0, i_col - 1] * Dx
    for i_row in range(1, n_diffs):
        X[i_row, i_row + 1:] = X[i_row - 1, i_row:-1]
    return F, NM, X
# ======================================================================
def _implicit_parameters_diffs(x):
    _sin = np.sin(x)
    _cos = np.cos(x)
    return np.array((_sin, _cos, -_sin, -_cos))
# ======================================================================
def implicit_parameters():
    info = ([], [], [])
    d = 4
    xa = 1.
    xb = 1.1
    Dx = xb - xa
    def rtol(v, ref):
        return abs(v - ref)
    DA = _implicit_parameters_diffs(xa)
    DB = _implicit_parameters_diffs(xb)

    info[2].append('True')
    info[0].append(DB[0])
    info[1].append(DB[1])

    info[2].append('Simple')
    info[0].append(DA[0] + DA[1]* Dx - DB[0])
    info[1].append(DA[1] + DA[2] * Dx - DB[1])

    info[2].append('Forward full')
    info[0].append(DA[0] + DA[1]* Dx + DA[2]/2*Dx**2 + DA[3]/6*Dx**3 - DB[0])
    info[1].append(DA[1] + DA[2]*Dx + DA[3]/2*Dx**2 - DB[1])

    info[2].append('Backward full')
    info[0].append(DA[0] + DB[1]* Dx + DB[2]/2*Dx**2 + DB[3]/6*Dx**3 - DB[0])
    info[1].append(DA[1] + DB[2]*Dx + DB[3]/2*Dx**2 - DB[1])

    info[2].append('Middle partial')
    info[0].append(DA[0] + DB[1] * Dx + DB[2]/2*Dx**2 - DB[0])
    info[1].append(DA[1] + (DA[2] + DB[2])/2 * Dx - DB[1])

    F, NM, X = _implicit_parameters_construct(d, Dx)
    print('F\n', F)
    print('NM\n', NM)
    print('X\n', X)

    
    info[2].append('Poly')
    for v in (0, 1):
        
        nv = NM[v, :d]
        print('nv\n', nv)
        mv = NM[v,d:]
        print('mv\n', mv)
        xv = X[v]
        print('xv\n', xv)
        # Inverting the rest of the M
        _NM = np.concatenate((NM[:v], NM[v+1:]))
        _X = np.concatenate((X[:v], X[v+1:]))
        _DB = np.concatenate((DB[:v], DB[v+1:]))


        print('_NM\n', _NM)
        _N = _NM[:,:d]
        _M = _NM[:,d:]
        IX = np.linalg.inv(np.eye(d-1) * _X[:,d:])
        IX = 1/ np.diag(_X[:,d:])
        print('IX\n', IX)
        IM = np.linalg.inv(_M)
        print('IM\n', IM)

        L = (mv * xv[d:]) * IX @ IM
        print('L\n', L)
        K = (nv * xv[:d] - L @ (_N * _X[:,:d])) / F

        print('K\n', K)

        # print()
        info[v].append(K @ DA + L @ _DB - DB[v])
        # print('DB_v\n', DB[v])
    
    print(*info[2])
    print(*info[0])
    print(*info[1])
# ======================================================================

# ======================================================================
main = get_main(__name__)