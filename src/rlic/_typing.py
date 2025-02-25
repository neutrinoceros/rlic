__all__ = ["f32", "f64", "ConvolveClosure", "FloatT"]
from typing import Literal, Protocol, TypeVar

from numpy import float32 as f32
from numpy import float64 as f64
from numpy.typing import NDArray

FloatT = TypeVar("FloatT", f32, f64)


# mypy (strict) flags that this typevar as "should be covariant",
# but pyright (strict) insists that it's really invariant, which is was I really
# mean here.
class ConvolveClosure(Protocol[FloatT]):  # type: ignore[misc]
    @staticmethod
    def closure(
        texture: NDArray[FloatT],
        u: NDArray[FloatT],
        v: NDArray[FloatT],
        kernel: NDArray[FloatT],
        iterations: int,
        uv_mode: Literal["velocity", "polarization"],
    ) -> NDArray[FloatT]: ...
