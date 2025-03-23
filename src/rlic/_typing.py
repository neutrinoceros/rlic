__all__ = ["f32", "f64", "ConvolveClosure", "FloatT"]
from typing import Literal, Protocol, TypeVar

from numpy import dtype, ndarray
from numpy import float32 as f32
from numpy import float64 as f64

FloatT = TypeVar("FloatT", f32, f64)


# mypy (strict) flags that this typevar as "should be covariant",
# but pyright (strict) insists that it's really invariant, which is what I really
# mean here.
class ConvolveClosure(Protocol[FloatT]):  # type: ignore[misc]
    @staticmethod
    def closure(
        texture: ndarray[tuple[int, int], dtype[FloatT]],
        u: ndarray[tuple[int, int], dtype[FloatT]],
        v: ndarray[tuple[int, int], dtype[FloatT]],
        kernel: ndarray[tuple[int], dtype[FloatT]],
        iterations: int,
        uv_mode: Literal["velocity", "polarization"],
    ) -> ndarray[tuple[int, int], dtype[FloatT]]: ...
