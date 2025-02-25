from typing import Literal, Protocol, TypeVar

from numpy import float32 as f32, float64 as f64
from numpy.typing import NDArray

FloatT = TypeVar("FloatT", f32, f64)


class ConvolveClosure(Protocol[FloatT]):
    @staticmethod
    def closure(
        texture: NDArray[FloatT],
        u: NDArray[FloatT],
        v: NDArray[FloatT],
        kernel: NDArray[FloatT],
        iterations: int,
        uv_mode: Literal["velocity", "polarization"],
    ) -> NDArray[FloatT]: ...
