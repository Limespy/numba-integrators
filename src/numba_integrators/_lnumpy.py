from typing import TYPE_CHECKING
from typing import TypeAlias

import numpy as np

# from . import _lnumba as nb
# ======================================================================
f16: TypeAlias = np.float16
f32: TypeAlias = np.float32
f64: TypeAlias = np.float64

i8: TypeAlias = np.int8
i16: TypeAlias = np.int16
i32: TypeAlias = np.int32
i64: TypeAlias = np.int64

u8: TypeAlias = np.uint8
u16: TypeAlias = np.uint16
u32: TypeAlias = np.uint32
u64: TypeAlias = np.uint64

if TYPE_CHECKING:
    ip: TypeAlias = i64
    up: TypeAlias = u64
else:
    ip = np.intp
    up = np.uintp
# ======================================================================
# TYPES
if TYPE_CHECKING:
    from typing import Any
    from typing import TypeAlias
    from typing import TypeVar
    from typing import TypeVarTuple
    from typing import Unpack

    Inty: TypeAlias = int | i8 | i16 | i32 | i64
    UInty: TypeAlias = u8 | u16 | u32 | u64 | up
    Floaty: TypeAlias = float | f16 | f32 | f64

    Index: TypeAlias = int | i32 | i64 | u32 | u64 | up
    T1 = TypeVar('T1', bound = Index)
    T2 = TypeVar('T2', bound = Index)
    T3 = TypeVar('T3', bound = Index)
    T4 = TypeVar('T4', bound = Index)
    T5 = TypeVar('T5', bound = Index)
    T6 = TypeVar('T6', bound = Index)
    T7 = TypeVar('T7', bound = Index)
    T8 = TypeVar('T8', bound = Index)

    # Shape: TypeAlias = tuple[T1, ...]
    Shape0: TypeAlias = tuple[()]
    Shape1: TypeAlias = tuple[T1]
    Shape2: TypeAlias = tuple[T1, T2]
    Shape3: TypeAlias = tuple[T1, T2, T3]
    Shape4: TypeAlias = tuple[T1, T2, T3, T4]
    Shape5: TypeAlias = tuple[T1, T2, T3, T4, T5]
    Shape6: TypeAlias = tuple[T1, T2, T3, T4, T5, T6]
    Shape7: TypeAlias = tuple[T1, T2, T3, T4, T5, T6, T7]

    _Scalar = TypeVar('_Scalar',
                      bound = np.generic, covariant = True,)

    ShapeType = TypeVar('ShapeType', bound = tuple[Any, ...])
    ShapeTuple = TypeVarTuple('ShapeTuple', default = Unpack[tuple[Any, ...]])

    Array: TypeAlias = np.ndarray[ShapeType, np.dtype[_Scalar]]

    Array0: TypeAlias = Array[Shape0, _Scalar]
    Array1: TypeAlias = Array[tuple[int], _Scalar]
    Array2: TypeAlias = Array[tuple[int, int], _Scalar]
    Array3: TypeAlias = Array[tuple[int, int, int], _Scalar]
    Array4: TypeAlias = Array[tuple[int, int, int, int], _Scalar]
    Array5: TypeAlias = Array[tuple[int, int, int, int, int], _Scalar]
    Array6: TypeAlias = Array[tuple[int, int, int, int, int, int], _Scalar]
    Array7: TypeAlias = Array[tuple[int, int, int, int, int, int, int], _Scalar]

    Number: TypeAlias = int | float

    _Array: TypeAlias = Array[tuple[*ShapeTuple], _Scalar] # type: ignore [misc,type-var]

    BoolArray: TypeAlias = _Array[*ShapeTuple, np.bool_] # type: ignore [type-var]

    # C64Array: TypeAlias = _Array[*ShapeTuple, c64]
    # C128Array: TypeAlias = _Array[*ShapeTuple, c128]
    # C256Array: TypeAlias = _Array[*ShapeTuple, c256]

    F16Array: TypeAlias = _Array[*ShapeTuple, f16] # type: ignore [type-var]
    F32Array: TypeAlias = _Array[*ShapeTuple, f32] # type: ignore [type-var]
    F64Array: TypeAlias = _Array[*ShapeTuple, f64] # type: ignore [type-var]
    # F128Array: TypeAlias = _Array[*ShapeTuple, f128]
    FArray: TypeAlias = F16Array | F32Array | F64Array

    I16Array: TypeAlias = _Array[*ShapeTuple, i16] # type: ignore [type-var]
    I32Array: TypeAlias = _Array[*ShapeTuple, i32] # type: ignore [type-var]
    I64Array: TypeAlias = _Array[*ShapeTuple, i64] # type: ignore [type-var]
    IArray: TypeAlias = I64Array | I32Array | I64Array

    U16Array: TypeAlias = _Array[*ShapeTuple, u16] # type: ignore [type-var]
    U32Array: TypeAlias = _Array[*ShapeTuple, u32] # type: ignore [type-var]
    U64Array: TypeAlias = _Array[*ShapeTuple, u64] # type: ignore [type-var]
    UPArray: TypeAlias = _Array[*ShapeTuple, up] # type: ignore [type-var]
    UArray: TypeAlias = U16Array | U32Array | U64Array | UPArray

    NumArray: TypeAlias = FArray | IArray | UArray

    Num: TypeAlias = NumArray | Number

    ScalarVar = TypeVar('ScalarVar', bound = np.generic)

    Start: TypeAlias = int
    Stop: TypeAlias = int
    Step: TypeAlias = int
    Length: TypeAlias = int
else:
    Callable = tuple

    Any = object

    Inty = UInty = Floaty = object

    Shape0 = Shape1 = Shape2 = Shape3 = Shape4 = Shape5 = type
    Shape6 = Shape7 = type
    Array = Array1 = Array2 = Array3 = Array4 = Array5 = Array6 = Array7 = tuple

    F16Array = F32Array = F64Array = FArray = Floaty = type
    I16Array = I32Array = I64Array = IArray = type
    U16Array = U32Array = U64Array = UArray = type
    NumArray = Num =  BoolArray = type
    ScalarVar = object
    Start = Stop = Step = Length = int
# ======================================================================
inf32 = f32(np.inf)
inf64 = f64(np.inf)
# # ======================================================================
# @nb.njit
# def rfrange(n: int, dtype: type = f64):
#     array = np.ones((n,), dtype)
#     for i in range(2, n):
#         array[i] = array[i-1] / i
#     return array
# # ======================================================================
# @nb.njit
# def rfrange_allocated(n: int,
#                       dtype: ScalarVar = f64,
#                       out: Array1[ScalarVar] = None):
#     if out is None:
#         array = np.ones((n,), dtype)
#     for i in range(2, n):
#         array[i] = array[i-1] / i
#     return array
# # ======================================================================
# @nb.njit
# def erange(value: float, n: int, dtype: type = f64):
#     array = np.ones((n,), dtype)
#     array[0] = value
#     for i in range(1, n):
#         array[i] = array[i-1] * value
#     return array
