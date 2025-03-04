from __future__ import annotations

__all__ = ["convolve"]

from typing import TYPE_CHECKING, Literal

import numpy as np

from rlic._core import convolve_f32, convolve_f64

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rlic._typing import ConvolveClosure, FloatT, f32, f64

_KNOWN_UV_MODES = ["velocity", "polarization"]
_SUPPORTED_DTYPES: list[np.dtype[np.floating]] = [
    np.dtype("float32"),
    np.dtype("float64"),
]


class _ConvolveF32:
    @staticmethod
    def closure(
        texture: NDArray[f32],
        u: NDArray[f32],
        v: NDArray[f32],
        kernel: NDArray[f32],
        iterations: int,
        uv_mode: Literal["velocity", "polarization"],
    ) -> NDArray[f32]:
        return convolve_f32(texture, u, v, kernel, iterations, uv_mode)


class _ConvolveF64:
    @staticmethod
    def closure(
        texture: NDArray[f64],
        u: NDArray[f64],
        v: NDArray[f64],
        kernel: NDArray[f64],
        iterations: int,
        uv_mode: Literal["velocity", "polarization"],
    ) -> NDArray[f64]:
        return convolve_f64(texture, u, v, kernel, iterations, uv_mode)


def convolve(
    texture: NDArray[FloatT],
    /,
    u: NDArray[FloatT],
    v: NDArray[FloatT],
    *,
    kernel: NDArray[FloatT],
    uv_mode: Literal["velocity", "polarization"] = "velocity",
    iterations: int = 1,
) -> NDArray[FloatT]:
    """2-dimensional line integral convolution.

    Apply Line Integral Convolution to a texture array, against a 2D flow
    (u, v) and via a 1D kernel.

    Arguments
    ---------
    texture: 2D numpy array (positional-only)
      Think of this as a tracer fluid. Random noise is a good input in the
      general case.

    u, v: 2D numpy arrays
      Represent the horizontal and vertical components of a vector field,
      respectively.

    kernel: 1D numpy array
      This is the convolution kernel. Think of it as relative weights along a
      portion of a field line. The first half of the array represent weights on
      the "past" part of a field line (with respect to a starting point), while
      the second line represents weights on the "future" part.

    uv_mode: 'velocity' (default), or 'polarization', keyword-only
      By default, the vector (u, v) field is assumed to be velocity-like, i.e.,
      its direction matters. With uv_mode='polarization', direction is
      effectively ignored.

    iterations: (positive) int (default: 1)
      Perform multiple iterations in a loop where the output array texture
      is fed back as the input to the next iteration.
      Looping is done at the native-code level.

    Returns
    -------
    2D numpy array
      The convolved texture. The dtype of the output array is the same as the
      input arrays.

    Raises
    ------
    TypeError
      If input arrays' dtypes are mismatched.
    ValueError:
      If non-sensical or unknown values are received.

    Notes
    -----
    All input arrays must have the same dtype, which can be either float32 or
    float64.

    Maximum performance is expected for C order arrays.

    With a kernel.size < 5, uv_mode='polarization' is effectively equivalent to
    uv_mode='velocity'. However, this is still a valid use case, so, no warning
    is emitted.

    It is recommended (but not required) to use odd-sized kernels, so that
    forward and backward passes are balanced.
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

    input_dtype = texture.dtype
    cc: ConvolveClosure[FloatT]
    # mypy ignores can be removed once Python 3.9 is dropped.
    if input_dtype == np.dtype("float32"):
        cc = _ConvolveF32  # type: ignore[assignment, unused-ignore] # pyright: ignore[reportAssignmentType]
    elif input_dtype == np.dtype("float64"):
        cc = _ConvolveF64  # type: ignore[assignment, unused-ignore] # pyright: ignore[reportAssignmentType]
    else:
        raise RuntimeError  # pragma: no cover
    return cc.closure(texture, u, v, kernel, iterations, uv_mode)
