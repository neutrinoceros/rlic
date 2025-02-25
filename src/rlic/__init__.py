"""Line Integral Convolution, implemented in Rust."""

from __future__ import annotations

__all__ = ["convolve"]

from typing import TYPE_CHECKING, Literal

import numpy as np

from rlic._core import convolve_f32, convolve_f64

if TYPE_CHECKING:
    from rlic._typing import (
        D1,
        D2,
        ConvolveClosure,
        FloatArray1D,
        FloatArray2D,
        FloatT,
        f32,
        f64,
    )

_KNOWN_UV_MODES = ["velocity", "polarization"]
_SUPPORTED_DTYPES: list[np.dtype[np.floating]] = [
    np.dtype("float32"),
    np.dtype("float64"),
]


class _ConvolveF32:
    @staticmethod
    def closure(
        texture: FloatArray2D[D1, D2, f32],
        u: FloatArray2D[D1, D2, f32],
        v: FloatArray2D[D1, D2, f32],
        kernel: FloatArray1D[int, f32],
        iterations: int,
        uv_mode: Literal["velocity", "polarization"],
    ) -> FloatArray2D[D1, D2, f32]:
        return convolve_f32(texture, u, v, kernel, iterations, uv_mode)


class _ConvolveF64:
    @staticmethod
    def closure(
        texture: FloatArray2D[D1, D2, f64],
        u: FloatArray2D[D1, D2, f64],
        v: FloatArray2D[D1, D2, f64],
        kernel: FloatArray1D[int, f64],
        iterations: int,
        uv_mode: Literal["velocity", "polarization"],
    ) -> FloatArray2D[D1, D2, f64]:
        return convolve_f64(texture, u, v, kernel, iterations, uv_mode)


def convolve(
    texture: FloatArray2D[D1, D2, FloatT],
    /,
    u: FloatArray2D[D1, D2, FloatT],
    v: FloatArray2D[D1, D2, FloatT],
    *,
    kernel: FloatArray1D[int, FloatT],
    iterations: int = 1,
    uv_mode: Literal["velocity", "polarization"] = "velocity",
) -> FloatArray2D[D1, D2, FloatT]:
    """2-dimensional line integral convolution.

    Apply Line Integral Convolution to a texture array, against a 2D flow
    (u, v) and via a 1D kernel.

    Arguments
    ---------
    texture: 2D numpy array (positional-only)
      Usually, random noise serves as input.

    u, v: 2D numpy arrays
      Represent the horizontal and vertical components of a vector field,
      respectively.

    kernel: 1D numpy array
      This is the convolution kernel.

    uv_mode: 'velocity' (default), or 'polarization', keyword-only
      By default, the vector (u, v) field is assumed to be velocity-like, i.e.,
      its direction matters. With uv_mode='polarization', direction is
      effectively ignored.

    iterations: (positive) int (default: 1)
      Perform multiple iterations in a loop where the output array texture
      is fed back as the input to the next iteration.
      Looping is done at the native-code level.

    Raises
    ------
    TypeError
      If input arrays' dtypes are mismatched.
    ValueError:
      If non-sensical or unknown values are received.

    Notes
    -----
    With a kernel.size < 5, uv_mode='polarization' is effectively equivalent to
    uv_mode='velocity'. However, this is still a valid use case, so, no warning
    is emitted.
    """
    if iterations < 0:
        raise ValueError(
            f"Invalid number of iterations: {iterations}\n"
            "Expected a strictly positive integer."
        )
    if iterations == 0:
        return texture.copy()

    if uv_mode not in _KNOWN_UV_MODES:
        raise ValueError(
            f"Invalid uv_mode {uv_mode!r}. Expected one of {_KNOWN_UV_MODES}"
        )

    dtype_error_expectations = (
        f"Expected texture, u, v and kernel with identical dtype, from {_SUPPORTED_DTYPES}. "
        f"Got {texture.dtype=}, {u.dtype=}, {v.dtype=}, {kernel.dtype=}"
    )

    input_dtypes = {arr.dtype for arr in (texture, u, v, kernel)}
    if unsupported_dtypes := input_dtypes.difference(_SUPPORTED_DTYPES):
        raise TypeError(
            f"Found unsupported data type(s): {list(unsupported_dtypes)}. "
            f"{dtype_error_expectations}"
        )

    if len(input_dtypes) != 1:
        raise TypeError(f"Data types mismatch. {dtype_error_expectations}")

    if texture.ndim != 2:
        raise ValueError(
            f"Expected an texture with exactly two dimensions. Got {texture.ndim=}"
        )
    if np.any(texture < 0):
        raise ValueError(
            "Found invalid texture element(s). Expected only positive values."
        )
    if u.shape != texture.shape or v.shape != texture.shape:
        raise ValueError(
            "Shape mismatch: expected texture, u and v with identical shapes. "
            f"Got {texture.shape=}, {u.shape=}, {v.shape=}"
        )

    if kernel.ndim != 1:
        raise ValueError(
            f"Expected a kernel with exactly one dimension. Got {kernel.ndim=}"
        )
    if kernel.size < 3:
        raise ValueError(f"Expected a kernel with size 3 or more. Got {kernel.size=}")
    if kernel.size > (max_size := min(texture.shape)):
        raise ValueError(
            f"{kernel.size=} exceeds the smallest dim of the texture ({max_size})"
        )
    if np.any(kernel < 0):
        raise ValueError(
            "Found invalid kernel element(s). Expected only positive values."
        )

    input_dtype = texture.dtype
    # mypy ignores can be removed once Python 3.9 is dropped.
    cc: ConvolveClosure[D1, D2, FloatT]
    if input_dtype == np.dtype("float32"):
        cc = _ConvolveF32  # type: ignore[assignment, unused-ignore] # pyright: ignore[reportAssignmentType]
    elif input_dtype == np.dtype("float64"):
        cc = _ConvolveF64  # type: ignore[assignment, unused-ignore] # pyright: ignore[reportAssignmentType]
    else:
        raise RuntimeError  # pragma: no cover
    return cc.closure(texture, u, v, kernel, iterations, uv_mode)
