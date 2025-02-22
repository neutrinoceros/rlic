from typing import Literal

from numpy.typing import NDArray

from rLIC._typing import FloatT

def convolve_iteratively(
    image: NDArray[FloatT],
    u: NDArray[FloatT],
    v: NDArray[FloatT],
    kernel: NDArray[FloatT],
    iterations: int = 1,
    uv_mode: Literal["velocity", "polarization"] = "velocity",
) -> NDArray: ...
