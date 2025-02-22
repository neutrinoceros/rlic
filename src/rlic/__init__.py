from rlic._core import convolve_iteratively
from typing import Literal

__all__ = ["convolve"]

_KNOWN_UV_MODES = ["velocity", "polarization"]
def convolve(
    image, u, v, *, kernel, iterations: int = 1, uv_mode:Literal["velocity", "polarization"]="velocity",
):
    if iterations < 0:
        raise ValueError(f"Invalid number of iterations {iterations}. "
                         "Expected a strictly positive integer.")
    if iterations == 0:
        return image.copy()
    if uv_mode not in _KNOWN_UV_MODES:
        raise ValueError(f"Invalid uv_mode {uv_mode!r}. "
                         f"Expected one of {_KNOWN_UV_MODES}")
    return convolve_iteratively(image, u, v, kernel, iterations, uv_mode)
