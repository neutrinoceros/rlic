from typing import Literal, Protocol, TypeVar

from numpy import float32 as f32, float64 as f64
from numpy.typing import NDArray

FloatT = TypeVar("FloatT", f32, f64)


class ConvolveClosure(Protocol[FloatT]):
    @staticmethod
    def closure(
        image: NDArray[FloatT],
        u: NDArray[FloatT],
        v: NDArray[FloatT],
        kernel: NDArray[FloatT],
        iterations: int = 1,
        uv_mode: Literal["velocity", "polarization"] = "velocity",
    ) -> NDArray[FloatT]: ...
