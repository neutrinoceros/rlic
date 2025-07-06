from typing import Literal

from numpy import dtype, ndarray
from numpy import float32 as f32
from numpy import float64 as f64

def convolve_f32(
    texture: ndarray[tuple[int, int], dtype[f32]],
    u: ndarray[tuple[int, int], dtype[f32]],
    v: ndarray[tuple[int, int], dtype[f32]],
    kernel: ndarray[tuple[int], dtype[f32]],
    uv_mode: Literal["velocity", "polarization"] = "velocity",
    iterations: int = 1,
) -> ndarray[tuple[int, int], dtype[f32]]: ...
def convolve_f64(
    texture: ndarray[tuple[int, int], dtype[f64]],
    u: ndarray[tuple[int, int], dtype[f64]],
    v: ndarray[tuple[int, int], dtype[f64]],
    kernel: ndarray[tuple[int], dtype[f64]],
    uv_mode: Literal["velocity", "polarization"] = "velocity",
    iterations: int = 1,
) -> ndarray[tuple[int, int], dtype[f64]]: ...
