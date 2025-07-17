from numpy import dtype, ndarray
from numpy import float32 as f32
from numpy import float64 as f64

from rlic._typing import BoundaryPair, UVMode

def convolve_f32(
    texture: ndarray[tuple[int, int], dtype[f32]],
    uv: tuple[
        ndarray[tuple[int, int], dtype[f32]],
        ndarray[tuple[int, int], dtype[f32]],
        UVMode,
    ],
    kernel: ndarray[tuple[int], dtype[f32]],
    boundaries: tuple[BoundaryPair, BoundaryPair],
    iterations: int = 1,
) -> ndarray[tuple[int, int], dtype[f32]]: ...
def convolve_f64(
    texture: ndarray[tuple[int, int], dtype[f64]],
    uv: tuple[
        ndarray[tuple[int, int], dtype[f64]],
        ndarray[tuple[int, int], dtype[f64]],
        UVMode,
    ],
    kernel: ndarray[tuple[int], dtype[f64]],
    boundaries: tuple[BoundaryPair, BoundaryPair],
    iterations: int = 1,
) -> ndarray[tuple[int, int], dtype[f64]]: ...
