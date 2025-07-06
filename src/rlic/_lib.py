from __future__ import annotations

__all__ = ["convolve"]

import sys
from typing import TYPE_CHECKING, Literal

import numpy as np

from rlic._core import convolve_f32, convolve_f64

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup  # pyright: ignore[reportUnreachable]

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy import dtype, ndarray
    from numpy import float32 as f32
    from numpy import float64 as f64

    F = TypeVar("F", f32, f64)

_KNOWN_UV_MODES = ["velocity", "polarization"]
_SUPPORTED_DTYPES: list[np.dtype[np.floating]] = [
    np.dtype("float32"),
    np.dtype("float64"),
]


def convolve(
    texture: ndarray[tuple[int, int], dtype[F]],
    /,
    u: ndarray[tuple[int, int], dtype[F]],
    v: ndarray[tuple[int, int], dtype[F]],
    *,
    kernel: ndarray[tuple[int], dtype[F]],
    uv_mode: Literal["velocity", "polarization"] = "velocity",
    iterations: int = 1,
) -> ndarray[tuple[int, int], dtype[F]]:
    """2-dimensional line integral convolution.

    Apply Line Integral Convolution to a texture array, against a 2D flow (u, v)
    and via a 1D kernel.

    Arguments
    ---------
    texture: 2D numpy array, positional-only
      Think of this as a tracer fluid. Random noise is a good input in the
      general case.

    u, v: 2D numpy arrays
      Represent the horizontal and vertical components of a vector field,
      respectively.

    kernel: 1D numpy array, keyword-only
      This is the convolution kernel. Think of it as relative weights along a
      portion of a field line. The first half of the array represent weights on
      the "past" part of a field line (with respect to a starting point), while
      the second line represents weights on the "future" part.

    uv_mode: 'velocity' (default), or 'polarization', keyword-only
      By default, the vector (u, v) field is assumed to be velocity-like, i.e.,
      its direction matters. With uv_mode='polarization', direction is
      effectively ignored.

    iterations: (positive) int (default: 1), keyword-only
      Perform multiple iterations in a loop where the output array texture is
      fed back as the input to the next iteration. Looping is done at the
      native-code level.

    Returns
    -------
    2D numpy array
      The convolved texture. The dtype of the output array is the same as the
      input arrays. The value returned is always a newly allocated array, even
      with `iterations=0`, in which case a copy of `texture` will be returned.

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

    Kernels cannot contain non-finite (infinite or NaN) values. Although
    unusual, negative values are allowed.

    No effort is made to avoid propagation of NaNs from the input texture.
    However, streamlines will be terminated whenever a pixel where either u or v
    contains a NaN is encountered.

    Infinite values in any input array are not special cased.

    This function is guaranteed to never mutate any input array, and always
    returns a newly allocated array. Thread-safety is thus trivially guaranteed.
    """
    exceptions: list[Exception] = []
    if iterations < 0:
        exceptions.append(
            ValueError(
                f"Invalid number of iterations: {iterations}\n"
                "Expected a strictly positive integer."
            )
        )

    if uv_mode not in _KNOWN_UV_MODES:
        exceptions.append(
            ValueError(
                f"Invalid uv_mode {uv_mode!r}. Expected one of {_KNOWN_UV_MODES}"
            )
        )

    dtype_error_expectations = (
        f"Expected texture, u, v and kernel with identical dtype, from {_SUPPORTED_DTYPES}. "
        f"Got {texture.dtype=}, {u.dtype=}, {v.dtype=}, {kernel.dtype=}"
    )

    input_dtypes = {arr.dtype for arr in (texture, u, v, kernel)}
    if unsupported_dtypes := input_dtypes.difference(_SUPPORTED_DTYPES):
        exceptions.append(
            TypeError(
                f"Found unsupported data type(s): {list(unsupported_dtypes)}. "
                f"{dtype_error_expectations}"
            )
        )

    if len(input_dtypes) != 1:
        exceptions.append(TypeError(f"Data types mismatch. {dtype_error_expectations}"))

    if texture.ndim != 2:
        exceptions.append(
            ValueError(
                f"Expected a texture with exactly two dimensions. Got {texture.ndim=}"
            )
        )
    if np.any(texture < 0):
        exceptions.append(
            ValueError(
                "Found invalid texture element(s). Expected only positive values."
            )
        )
    if u.shape != texture.shape or v.shape != texture.shape:
        exceptions.append(
            ValueError(
                "Shape mismatch: expected texture, u and v with identical shapes. "
                f"Got {texture.shape=}, {u.shape=}, {v.shape=}"
            )
        )

    if kernel.ndim != 1:
        exceptions.append(
            ValueError(
                f"Expected a kernel with exactly one dimension. Got {kernel.ndim=}"
            )
        )
    if np.any(~np.isfinite(kernel)):
        exceptions.append(ValueError("Found non-finite value(s) in kernel."))

    if len(exceptions) == 1:
        raise exceptions[0]
    elif exceptions:
        raise ExceptionGroup("Invalid inputs were received.", exceptions)

    if iterations == 0:
        return texture.copy()

    input_dtype = texture.dtype
    # mypy ignores can be removed once Python 3.9 is dropped.
    # https://github.com/numpy/numpy/issues/28572
    if input_dtype == np.dtype("float32"):
        return convolve_f32(texture, u, v, kernel, uv_mode, iterations)  # type: ignore[arg-type, return-value, unused-ignore] # pyright: ignore[reportArgumentType, reportReturnType]
    elif input_dtype == np.dtype("float64"):
        return convolve_f64(texture, u, v, kernel, uv_mode, iterations)  # type: ignore[arg-type, return-value, unused-ignore] # pyright: ignore[reportArgumentType, reportReturnType]
    else:
        raise RuntimeError  # pragma: no cover
