from typing import Literal

from numpy import float32 as f32
from numpy import float64 as f64
from numpy.typing import NDArray

def convolve_f32(
    texture: NDArray[f32],
    u: NDArray[f32],
    v: NDArray[f32],
    kernel: NDArray[f32],
    iterations: int = 1,
    uv_mode: Literal["velocity", "polarization"] = "velocity",
) -> NDArray[f32]: ...
def convolve_f64(
    texture: NDArray[f64],
    u: NDArray[f64],
    v: NDArray[f64],
    kernel: NDArray[f64],
    iterations: int = 1,
    uv_mode: Literal["velocity", "polarization"] = "velocity",
) -> NDArray[f64]: ...
