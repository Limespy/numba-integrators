from typing import TYPE_CHECKING
from warnings import filterwarnings

import numba as nb
import numpy as np
from limedev.CLI import get_main
# ======================================================================
if TYPE_CHECKING:
    from numba_integrators._lnumpy import F64Array, UPArray
else:
    F64Array = UPArray = tuple
# ======================================================================
filterwarnings('ignore', category = nb.errors.NumbaPerformanceWarning)
# ======================================================================
@nb.njit
def nb_solve(A, b):
    return np.linalg.solve(A,b)
# ======================================================================
@nb.njit
def pivot[N: int](A: F64Array[N, N]
                    ) -> tuple[F64Array[N, N], UPArray[N]]:
    n = A.shape[0]
    piv = np.zeros(n, np.uintp)
    for j in range(n-1):
        maxpivot = abs(A[j,j])
        maxpivotrow = np.uintp(j)
        # find max
        for i_row in range(j + 1, n):
            _abs_pivot = abs(A[i_row,j])
            # print(_abs_pivot, i_row)
            if _abs_pivot > maxpivot:
                maxpivot = _abs_pivot
                maxpivotrow = np.uintp(i_row)
        # print(maxpivotrow)
        piv[j] = maxpivotrow

        # Swap rows
        # if j != maxpivotrow:
        for i_col in range(n):
            tmp = A[j, i_col]
            A[j, i_col] = A[maxpivotrow, i_col]
            A[maxpivotrow, i_col] = tmp
    end = np.uintp(n-1)
    piv[end] = end
    return A, piv
# ======================================================================
@nb.njit(nb.types.Tuple((nb.float64[:, ::1], nb.uintp[::1])
                        )(nb.float64[:, ::1]),
            fastmath = False)
def pivot_fast[N: int](A: F64Array[N, N]
                    ) -> tuple[F64Array[N, N], UPArray[N]]:
    n = np.uintp(A.shape[0])
    piv = np.zeros(n, np.uintp)
    _0_index = np.uintp(0)
    _1_index = np.uintp(1)
    j = _0_index
    end = n - _1_index
    while j < end:
        maxpivot = abs(A[j,j])
        maxpivotrow = j
        # find max
        i_row = j + _1_index
        while i_row < n:
            _abs_pivot = abs(A[i_row,j])
            if _abs_pivot > maxpivot:
                maxpivot = _abs_pivot
                maxpivotrow = i_row
            i_row += _1_index
        piv[j] = maxpivotrow

        # Swap rows
        i_col = _0_index
        while i_col < n:
            tmp = A[j, i_col]
            A[j, i_col] = A[maxpivotrow, i_col]
            A[maxpivotrow, i_col] = tmp
            i_col += _1_index
        j += _1_index

    piv[end] = end
    return A, piv
# ======================================================================
@nb.njit
def nb_lu_factor_base(A):
    n = A.shape[0]
    A, piv = pivot_fast(A)
    LU = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            _sum = 0
            if i <= j:
                for k in range(i):
                    _sum += LU[i,k] * LU[k,j]

                LU[i,j] = A[i,j] - _sum

            else:
                for k in range(j):
                    _sum += LU[i,k] * LU[k,j]

                LU[i,j] = (A[i,j] - _sum)

    return LU, piv
# ======================================================================
@nb.njit
def nb_lu_factor(A):
    n = A.shape[0]
    A, piv = pivot_fast(A)

    LU = A

    _1_LU_i_col_i_col = 1. / LU[0,0]

    for i_row in range(1, n):
        LU[i_row,0] *= _1_LU_i_col_i_col

    for i_col in range(1, n):

        # print('col', i_col)
        for i_row in range(1, i_col + 1):
            _sum = LU[i_row, 0] * LU[0, i_col]
            for k in range(1, i_row):
                # print('a access', (i_row, k, LU[i_row,k]), (k,i_col, LU[k,i_col]))
                # _sum += tmp[i_row,k]
                _sum += LU[i_row,k] * LU[k,i_col]
                # _sum += LU[i_row,k] * tmp[k]
            LU[i_row,i_col] -= _sum
            # print('a', (i_row, i_col), _sum)

        _1_LU_i_col_i_col = 1. / LU[i_col,i_col]
        for i_row in range(i_col + 1, n):
            _sum = LU[i_row,0] * LU[0, i_col]
            for k in range(1,i_col):
                # print('b access', (i_row,k, LU[i_row,k]), (k,i_col, LU[k,i_col]))
                # _sum += tmp[i_row,k]
                _sum += LU[i_row,k] * LU[k,i_col]
                # _sum += LU[i_row,k] * tmp[k]
            # print('b', (i_row, i_col), _sum)
            LU[i_row, i_col] -= _sum
            LU[i_row, i_col] *=  _1_LU_i_col_i_col

    return LU, piv
# ======================================================================
@nb.njit
def nb_lu_factor_row(A):
    n = A.shape[0]
    A, piv = pivot_fast(A)
    LU = A
    _1_LU_0_0 = 1. / LU[0,0]

    for i_col in range(1, n):
        LU[0, i_col] *= _1_LU_0_0

    for i_row in range(1, n):

        # print('col', _)
        for i_col in range(1, i_row + 1):
            _sum = LU[0, i_col] * LU[i_row, 0]
            for k in range(1, i_col):
                # print('a access', (i_col, k, LU[k, i_col]), (i_row, k, LU[i_row, k]))
                # _sum += tmp[k, i_col]
                _sum += LU[k, i_col] * LU[i_row, k]
                # _sum += LU[k, i_col] * tmp[k]
            LU[i_row, i_col] -= _sum
            # print('a', (i_col, i_row), _sum)

        _1_LU_i_col_i_col = 1. / LU[i_row,i_row]
        for i_col in range(i_row + 1, n):
            _sum = LU[0, i_col] * LU[i_row, 0]
            for k in range(1,i_row):
                # print('b access', (k, i_col, LU[k, i_col]), (i_row, k, LU[i_row, k]))
                # _sum += tmp[k, i_col]
                _sum += LU[k, i_col] * LU[i_row, k]
                # _sum += LU[k, i_col] * tmp[k]
            # print('b', (i_col, i_row), _sum)
            LU[i_row, i_col] -= _sum
            LU[i_row, i_col] *=  _1_LU_i_col_i_col

    return LU.T, piv
# ======================================================================
@nb.njit(fastmath = False)
def _nb_lu_factor_recurse(A):
    n = A.shape[0]
    if n == 1:
        half = n//2
        A11 = A[:half, :half]
        A21 = A[half:, :half]
        A12 = A[:half, half:]
        A22 = A[half:, half:]

        # 1. If m = 1 then factor (that is, perform pivoting and scaling)
        #
        # P_1 @ [A_11;A_21] = [L_11, L_21] @ U_11
        #
        # and return.
        pivot_fast()
        return
    # 2. Else, recursively factor
    #
    # P_1 @ [A_11\\A_21] = [L_11\\ L_21] @ U_11
    #
    # 3. Permute
    #
    # [A_12'\\ A_22'] <- P_1 @ [A_12\\A_22]
    #
    # 4. Solve the triangular system L11 @ U12 = A_12′ for U_12.
    #
    # 5. A_22'' <- A_22' - L_21 @ U_12
    #
    # 6. Recursively factor P_2 @ A_22'' = L_22 @ U_22
    #
    # 7. Permute L_21' <- P_2 @ L_21
    #
    # 8. Return
    #
    # P_2 @ P_1 @ [A_11, A_12\\A_21, A_22]
    #   = [L_11, 0 \\ L_21', L_22] @ [U_11, U_12\\0, U_22]

    A, piv = pivot_fast(A[:,:n//2])
# ======================================================================
@nb.njit(nb.types.Tuple((nb.float64[:, ::1], nb.uintp[::1])
                        )(nb.float64[:, ::1]),
            fastmath = False, parallel = False)
def nb_lu_factor_fast(A):
    n = A.shape[0]
    A, piv = pivot_fast(A)

    LU = A

    _1_LU_0_0 = 1. / LU[0,0]

    for i_row in range(1, n):
        LU[i_row,0] *= _1_LU_0_0

    for i_col in range(1, n-1):

        for j in nb.prange(i_col, n):
            _sum = LU[i_col,0] * LU[0,j]
            for k in range(1, i_col):
                _sum += LU[i_col,k] * LU[k,j]
            LU[i_col,j] -= _sum

        # for k in range(i_col):
        #     LU[i_col,i_col:n] -= LU[i_col,k] * LU[k,i_col:n]

        # LU[i_col,i_col:] -= LU[i_col,:i_col] @ LU[:i_col,i_col:]
        # # print('col', i_col)
        # for i_row in range(1, i_col + 1):
        #     _sum = LU[i_row, 0] * LU[0, i_col]
        #     for k in range(1, i_row):
        #         # print('a access', (i_row, k, LU[i_row,k]), (k,i_col, LU[k,i_col]))
        #         # _sum += tmp[i_row,k]
        #         _sum += LU[i_row,k] * LU[k,i_col]
        #         # _sum += LU[i_row,k] * tmp[k]
        #     LU[i_row,i_col] -= _sum
        #     # print('a', (i_row, i_col), _sum)

        _1_LU_i_col_i_col = 1. / LU[i_col,i_col]
        for i_row in nb.prange(i_col + 1, n):
            _sum = LU[i_row, 0] * LU[0, i_col]
            for k in range(1, i_col):
                # print('b access', (i_row,k, LU[i_row,k]), (k,i_col, LU[k,i_col]))
                _sum += LU[i_row, k] * LU[k, i_col]
            # print('b', (i_row, i_col), _sum)
            LU[i_row, i_col] -= _sum
            LU[i_row, i_col] *=  _1_LU_i_col_i_col
            # LU[k+1:,k] -= LU[k+1:,:k] @ LU[:k,k]

    _sum = LU[-1,0] * LU[0,-1]
    for k in range(1, n-1):
        _sum += LU[-1,k] * LU[k,-1]
    LU[-1,-1] -= _sum
    return LU, piv
# ======================================================================
@nb.njit(parallel = False)
def nb_lu_factor_matrix(A):
    r"""Compute the Doolittle LU factorization of A.

    Sums like $\sum_{s=1}^{k-1} l_{k,s} u_{s,j}$ are done as matrix
    products; in the above case, row matrix L[k, 1:k-1] by column matrix
    U[1:k-1,j] gives the sum for a give j, and row matrix L[k, 1:k-1] by
    matrix U[1:k-1,k:n] gives the relevant row vector.
    """
    n = A.shape[0]  # len() gives the number of rows in a 2D array.
    A, piv = pivot_fast(A)

    LU = A
    # Column and row 1 (i.e Python index 0) are special:
    _1_LU_0_0 = 1. / LU[0,0]
    LU[1:,0] *= _1_LU_0_0

    for k in range(1, n-1):
        LU[k,k:] -= LU[k,:k] @ LU[:k,k:]

        LU[k+1:,k] -= LU[k+1:,:k] @ LU[:k,k]
        _1_LU_k_k = 1. / LU[k,k]
        LU[k+1:,k] *= _1_LU_k_k

    # The last row (index "-1") is special: nothing to do for L
    LU[-1,-1] -= LU[-1,:-1] @ LU[:-1,-1]

    return LU, piv
# ======================================================================
def time_matrix_solve(n_points: int = 15):
    from collections import defaultdict
    from time import perf_counter

    from matplotlib import pyplot as plt
    from scipy import linalg
    import numba as nb



    rng = np.random.default_rng(1234)
    sqrt2 = 2**0.5

    lengths = np.zeros(n_points, np.uint32)
    multiplier = 2.
    for index in range(n_points):
        lengths[index] = round(multiplier)
        multiplier *= sqrt2
    elements = lengths*lengths
    times = defaultdict(list)


    A = np.array(((1, 5, -2, 1),
                  (-2, -1, 5, -1),
                  (3, -2, 4, 3),
                  (4, 3, 7, 4)), np.float64)

    print(*pivot(A.copy()))
    print(*pivot_fast(A.copy()))


    # print(*nb_lu_factor(A.copy()))
    # print(*nb_lu_factor_fast(A.copy()))

    # _A = A.copy()
    print(*linalg.lu_factor(A.copy()))
    # print(f'{p}\n{l}\n{u}\n')

    b = np.array((3.,4.,5.,6.))

    ref = linalg.lu_solve(linalg.lu_factor(A.copy()), b)

    # print('base', linalg.lu_solve(nb_lu_factor_base(A.copy()), b)
    #       - ref)
    print('normal', linalg.lu_solve(nb_lu_factor(A.copy()), b)
          - ref)

    print('fast', linalg.lu_solve(nb_lu_factor_fast(A.copy()), b)
          - ref)
    print('matrix', linalg.lu_solve(nb_lu_factor_matrix(A.copy()), b)
          - ref)

    # print(*linalg.lu_factor(A.copy()))
    # return
    for length, n_elements in zip(lengths, elements):
        print(length)
        scale = 1./n_elements
        A = rng.random((length, length))
        b = rng.random((length, ))

        # t0 = perf_counter()
        # np.linalg.solve(A,b)
        # times['np solve'].append((perf_counter() - t0) * scale)

        # nb_solve(A,b)
        # nb_solve(A,b)
        # nb_solve(A,b)

        # t0 = perf_counter()
        # nb_solve(A,b)
        # times['nb solve'].append((perf_counter() - t0) * scale)

        # t0 = perf_counter()
        # linalg.solve(A,b)
        # times['scipy solve'].append((perf_counter() - t0) * scale)

        # t0 = perf_counter()
        # linalg.inv(A)
        # times['scipy inv'].append((perf_counter() - t0) * scale)

        ## Pivoting

        pivot(A.copy())
        pivot(A.copy())
        pivot(A.copy())

        _A = A.copy()
        t0 = perf_counter()
        pivot(_A)
        times['pivot'].append((perf_counter() - t0) * scale)


        pivot_fast(A.copy())
        pivot_fast(A.copy())
        pivot_fast(A.copy())

        _A = A.copy()
        t0 = perf_counter()
        pivot_fast(_A)
        times['pivot fast'].append((perf_counter() - t0) * scale)

        # _A = A.copy()
        # t0 = perf_counter()
        # LU, piv = linalg.lu_factor(_A, overwrite_a = True)
        # times['scipy lu factor overwrite'].append((perf_counter() - t0) * scale)

        # _A = A.copy()
        # t0 = perf_counter()
        # LU, piv = linalg.lu_factor(_A)
        # times['scipy lu factor'].append((perf_counter() - t0) * scale)

        # nb_lu_factor_base(A.copy())
        # nb_lu_factor_base(A.copy())
        # nb_lu_factor_base(A.copy())

        # _A = A.copy()
        # t0 = perf_counter()
        # nb_lu_factor_base(_A)
        # times['nb lu factor base'].append((perf_counter() - t0) * scale)

        # nb_lu_factor(A.copy())
        # nb_lu_factor(A.copy())
        # nb_lu_factor(A.copy())

        # _A = A.copy()
        # t0 = perf_counter()
        # nb_lu_factor(_A)
        # times['nb lu factor'].append((perf_counter() - t0) * scale)

        # nb_lu_factor_fast(A.copy())
        # nb_lu_factor_fast(A.copy())
        # nb_lu_factor_fast(A.copy())

        # _A = A.copy()
        # t0 = perf_counter()
        # nb_lu_factor_fast(_A)
        # times['nb lu factor fast'].append((perf_counter() - t0) * scale)

        # nb_lu_factor_matrix(A.copy())

        # _A = A.copy()
        # t0 = perf_counter()
        # nb_lu_factor_matrix(_A)
        # times['nb lu factor matrix'].append((perf_counter() - t0) * scale)

        # t0 = perf_counter()
        # linalg.lu_solve((LU, piv), b)
        # times['scipy lu solve'].append((perf_counter() - t0) * scale)
    del A
    del b
    plt.ion()
    for label, _times in times.items():
        plt.loglog(elements, _times, label = label)
    plt.ylabel('time / elements')
    plt.xlabel('elements')
    plt.legend()
    plt.grid()
    plt.show()
    input()
# ======================================================================
main = get_main(__name__)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    raise SystemExit(main())
