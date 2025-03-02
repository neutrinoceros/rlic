__all__ = ["f32", "f64", "ConvolveClosure", "FloatT"]
from typing import Literal, Protocol, TypeAlias, TypeVar

from numpy import dtype, ndarray
from numpy import float32 as f32
from numpy import float64 as f64

FloatT = TypeVar("FloatT", f32, f64)

D1 = TypeVar("D1", bound=int)
D2 = TypeVar("D2", bound=int)


FloatArray1D: TypeAlias = ndarray[tuple[D1], dtype[FloatT]]
FloatArray2D: TypeAlias = ndarray[tuple[D1, D2], dtype[FloatT]]

# mypy (strict) flags that this typevar as "should be covariant",
# but pyright (strict) insists that it's really invariant, which is was I really
# mean here.
class ConvolveClosure(Protocol[D1, D2, FloatT]): # type: ignore[misc]
    @staticmethod
    def closure(
        texture: FloatArray2D[D1, D2, FloatT],
        u: FloatArray2D[D1, D2, FloatT],
        v: FloatArray2D[D1, D2, FloatT],
        kernel: FloatArray1D[int, FloatT],
        iterations: int,
        uv_mode: Literal["velocity", "polarization"],
    ) -> FloatArray2D[D1, D2, FloatT]: ...
