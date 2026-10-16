from typing import TypeVar

from numpy import dtype, ndarray
from numpy import float32 as f32
from numpy import float64 as f64

from rlic._boundaries import BoundaryStr
from rlic._typing import Pair, UVMode

I = TypeVar("I", bound=int)
J = TypeVar("J", bound=int)
F = TypeVar("F", f32, f64)

IJArray = ndarray[tuple[I, J], dtype[F]]

def convolve_f32(
    texture: IJArray[I, J, f32],
    uv: tuple[
        IJArray[I, J, f32],
        IJArray[I, J, f32],
        UVMode,
    ],
    kernel: ndarray[tuple[int], dtype[f32]],
    boundaries: Pair[Pair[BoundaryStr]],
    iterations: int = 1,
) -> IJArray[I, J, f32]: ...
def convolve_f64(
    texture: IJArray[I, J, f64],
    uv: tuple[
        IJArray[I, J, f64],
        IJArray[I, J, f64],
        UVMode,
    ],
    kernel: ndarray[tuple[int], dtype[f64]],
    boundaries: Pair[Pair[BoundaryStr]],
    iterations: int = 1,
) -> IJArray[I, J, f64]: ...
def equalize_histogram_f32(
    image: IJArray[I, J, f32],
    nbins: int,
) -> IJArray[I, J, f32]: ...
def equalize_histogram_f64(
    image: IJArray[I, J, f64],
    nbins: int,
) -> IJArray[I, J, f64]: ...
def equalize_histogram_sliding_tile_f32(
    image: IJArray[I, J, f32],
    nbins: int,
    tile_shape: Pair[int],
) -> IJArray[I, J, f32]: ...
def equalize_histogram_sliding_tile_f64(
    image: IJArray[I, J, f64],
    nbins: int,
    tile_shape: Pair[int],
) -> IJArray[I, J, f64]: ...
def equalize_histogram_tile_interpolation_f32(
    image: IJArray[I, J, f32],
    nbins: int,
    tile_shape: Pair[int],
) -> IJArray[I, J, f32]: ...
def equalize_histogram_tile_interpolation_f64(
    image: IJArray[I, J, f64],
    nbins: int,
    tile_shape: Pair[int],
) -> IJArray[I, J, f64]: ...
