__all__ = ["convolve"]

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rlic._core import convolve_iteratively
from rlic._typing import FloatT

_KNOWN_UV_MODES = ["velocity", "polarization"]


def convolve(
    image: NDArray[FloatT],
    u: NDArray[FloatT],
    v: NDArray[FloatT],
    *,
    kernel: NDArray[FloatT],
    iterations: int = 1,
    uv_mode: Literal["velocity", "polarization"] = "velocity",
):
    if kernel.size < 3:
        raise ValueError(f"Expected a kernel with size 3 or more. Got {kernel.size=}")
    if kernel.size > (max_size := min(image.shape)):
        raise ValueError(
            f"{kernel.size=} exceeds the smallest dim of the image ({max_size})"
        )
    if np.any(kernel < 0):
        raise ValueError(
            "Found invalid kernel element(s). Expected only positive values."
        )

    if iterations < 0:
        raise ValueError(
            f"Invalid number of iterations {iterations}. "
            "Expected a strictly positive integer."
        )
    if iterations == 0:
        return image.copy()
    if uv_mode not in _KNOWN_UV_MODES:
        raise ValueError(
            f"Invalid uv_mode {uv_mode!r}. Expected one of {_KNOWN_UV_MODES}"
        )
    return convolve_iteratively(image, u, v, kernel, iterations, uv_mode)
