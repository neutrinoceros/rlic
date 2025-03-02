from typing import Literal

from numpy import float32 as f32
from numpy import float64 as f64

from rlic._typing import D1, D2, FloatArray1D, FloatArray2D

def convolve_f32(
    texture: FloatArray2D[D1, D2, f32],
    u: FloatArray2D[D1, D2, f32],
    v: FloatArray2D[D1, D2, f32],
    kernel: FloatArray1D[int, f32],
    iterations: int,
    uv_mode: Literal["velocity", "polarization"],
) -> FloatArray2D[D1, D2, f32]: ...
def convolve_f64(
    texture: FloatArray2D[D1, D2, f64],
    u: FloatArray2D[D1, D2, f64],
    v: FloatArray2D[D1, D2, f64],
    kernel: FloatArray1D[int, f64],
    iterations: int,
    uv_mode: Literal["velocity", "polarization"],
) -> FloatArray2D[D1, D2, f64]: ...
