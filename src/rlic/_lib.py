from __future__ import annotations

__all__ = [
    "convolve",
    "equalize_histogram",
]

import sys
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from rlic._boundaries import BoundarySet
from rlic._core import (
    convolve_f32,
    convolve_f64,
    equalize_histogram_f32,
    equalize_histogram_f64,
    equalize_histogram_sliding_tile_f32,
    equalize_histogram_sliding_tile_f64,
    equalize_histogram_tile_interpolation_f32,
    equalize_histogram_tile_interpolation_f64,
)
from rlic._histeq import (
    SUPPORTED_AHE_KINDS,
    SlidingTile,
    TileInterpolation,
)
from rlic._typing import UNSET, UnsetType

if sys.version_info >= (3, 11):
    from typing import assert_never  # pyright: ignore[reportUnreachable]
else:
    from exceptiongroup import ExceptionGroup  # pyright: ignore[reportUnreachable]
    from typing_extensions import assert_never  # pyright: ignore[reportUnreachable]

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    from numpy import dtype, ndarray
    from numpy import float32 as f32
    from numpy import float64 as f64

    from rlic._boundaries import BoundarySpec, BoundaryStr
    from rlic._histeq import Strategy, StrategySpec
    from rlic._typing import Pair, UVMode

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
    uv_mode: UVMode = "velocity",
    boundaries: BoundarySpec = "closed",
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

    boundaries: 'closed' (default), 'periodic', or a dict with keys 'x' and 'y',
      and values are either of these strings, or 2-tuples (left, right) thereof.
      This controls what boundary conditions are applied during streamline
      integration. It is possible to specify left and right boundaries
      independently along either directions, where x and y are the directions
      parallel to the u and v vector field components, respectively.
      A single string is also accepted as a shorthand for setting all boundaries
      to the same type.
      A 'periodic' boundary can only be combined with an identical one on
      the opposite side.

      .. versionadded: 0.5.0

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
    ExceptionGroup:
      If more than a single exception is detected, they'll all be raised as
      an exception group.

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

    if (bs := BoundarySet.from_spec(boundaries)) is None:
        exceptions.append(TypeError(f"Invalid boundary specification {boundaries}"))
    else:
        exceptions.extend(bs.collect_exceptions())

    if len(exceptions) == 1:
        raise exceptions[0]
    elif exceptions:
        raise ExceptionGroup("Invalid inputs were received.", exceptions)

    bs = cast("BoundarySet", bs)
    if iterations == 0:
        return texture.copy()

    input_dtype = texture.dtype
    retf: Callable[
        [
            ndarray[tuple[int, int], dtype[F]],
            tuple[
                ndarray[tuple[int, int], dtype[F]],
                ndarray[tuple[int, int], dtype[F]],
                UVMode,
            ],
            ndarray[tuple[int], dtype[F]],
            Pair[Pair[BoundaryStr]],
            int,
        ],
        ndarray[tuple[int, int], dtype[F]],
    ]
    # about type: and pyright: comments:
    # https://github.com/numpy/numpy/issues/28572
    if input_dtype == np.dtype("float32"):
        retf = convolve_f32  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
    elif input_dtype == np.dtype("float64"):
        retf = convolve_f64  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
    else:
        raise RuntimeError

    return retf(texture, (u, v, uv_mode), kernel, (bs.x, bs.y), iterations)


def _resolve_nbins(nbins: int | Literal["auto"], shape: Pair[int]) -> int:
    if nbins == "auto":
        npix = shape[0] * shape[1]
        return min(npix, 256)
    else:
        return nbins


def equalize_histogram(
    image: ndarray[tuple[int, int], dtype[F]],
    /,
    *,
    nbins: int | Literal["auto"] = "auto",
    boundaries: BoundarySpec = "closed",
    adaptive_strategy: StrategySpec | None = None,
    contrast_limitation: None = None,
) -> ndarray[tuple[int, int], dtype[F]]:
    """Equalize histogram of a gray-scale image.

    Parameters
    ----------
    image : 2D array, positional only
      The input gray-scale image.

    nbins: int or 'auto', keyword-only
      number of bins to use in histograms
      By default ('auto'), this is set to 256 or the number maximum of pixels in a tile,
      whichever is smallest.
      Reduce this number for faster computations.
      Increase it to improve the overall contrast of the result.

    boundaries: 'closed' (default), 'periodic', or a dict with keys 'x' and 'y',
                and values are either of these strings, or 2-tuples (left, right)
                thereof. Keyword-only

      Only 'closed' boundaries are accepted at the moment.
      https://github.com/neutrinoceros/rlic/issues/303

    adaptive_strategy: None (default) or a sliding-tile specification, keyword-only
      not implemented
      https://github.com/neutrinoceros/rlic/issues/301
      https://github.com/neutrinoceros/rlic/issues/302

    contrast_limitation: None, keyword-only
      not implemented
      https://github.com/neutrinoceros/rlic/issues/304

    Returns
    -------
    2D array
        The processed image with values normalized to the [0, 1] interval.
    """
    ALL_CLOSED = BoundarySet(x=("closed", "closed"), y=("closed", "closed"))
    if BoundarySet.from_spec(boundaries) != ALL_CLOSED:
        raise NotImplementedError

    if contrast_limitation is not None:
        raise NotImplementedError  # type: ignore

    if image.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"Found unsupported data type: {image.dtype}. "
            f"Expected of of {_SUPPORTED_DTYPES}."
        )

    input_dtype = image.dtype
    if adaptive_strategy is None:
        histeq: Callable[
            [ndarray[tuple[int, int], dtype[F]], int],
            ndarray[tuple[int, int], dtype[F]],
        ]
        if input_dtype == np.dtype("float32"):
            histeq = equalize_histogram_f32  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
        elif input_dtype == np.dtype("float64"):
            histeq = equalize_histogram_f64  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
        else:
            raise AssertionError
        nbins = _resolve_nbins(nbins, image.shape)
        return histeq(image, nbins)

    ahe_type: type[Strategy]
    match adaptive_strategy.get("kind", UNSET):
        case "sliding-tile":
            ahe_type = SlidingTile
        case "tile-interpolation":
            ahe_type = TileInterpolation
        case str() as unknown:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(  # pyright: ignore[reportUnreachable]
                f"Unknown strategy kind {unknown!r}. "
                f"Expected one of {sorted(SUPPORTED_AHE_KINDS)}"
            )
        case UnsetType():  # pyright: ignore[reportUnnecessaryComparison]
            raise TypeError("adaptive_strategy is missing a 'kind' key.")  # pyright: ignore[reportUnreachable]  s
        case _ as invalid:  # pyright: ignore[reportUnnecessaryComparison]
            raise TypeError(  # pyright: ignore[reportUnreachable]
                f"Invalid strategy kind {invalid!r} with type {type(invalid)}. "
                f"Expected one of {sorted(SUPPORTED_AHE_KINDS)}"
            )

    strat = ahe_type.from_spec(adaptive_strategy)
    ts = strat.resolve_tile_shape(image.shape)
    nbins = _resolve_nbins(nbins, ts)

    pad_width = strat.resolve_pad_width(image.shape)
    pimage = np.pad(image, pad_width=pad_width, mode="reflect")

    match strat:
        case SlidingTile():
            histeq_st: Callable[
                [ndarray[tuple[int, int], dtype[F]], int, Pair[int]],
                ndarray[tuple[int, int], dtype[F]],
            ]
            if input_dtype == np.dtype("float32"):
                histeq_st = equalize_histogram_sliding_tile_f32  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
            elif input_dtype == np.dtype("float64"):
                histeq_st = equalize_histogram_sliding_tile_f64  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
            else:
                raise AssertionError
            res = histeq_st(pimage, nbins, ts)  # type: ignore[arg-type]
        case TileInterpolation():
            histeq_ti: Callable[
                [ndarray[tuple[int, int], dtype[F]], int, Pair[int]],
                ndarray[tuple[int, int], dtype[F]],
            ]
            if input_dtype == np.dtype("float32"):
                histeq_ti = equalize_histogram_tile_interpolation_f32  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
            elif input_dtype == np.dtype("float64"):
                histeq_ti = equalize_histogram_tile_interpolation_f64  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
            else:
                raise AssertionError
            res = histeq_ti(pimage, nbins, ts)  # type: ignore[arg-type]
        case _ as unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)

    # unpad result
    return res[pad_width[0][0] : -pad_width[0][1], pad_width[1][0] : -pad_width[1][1]]  # type: ignore[return-value]
