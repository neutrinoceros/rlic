from lick_core._core import convolve_loop

__all__ = ["convolve"]

def convolve(
    image, u, v, *, kernel, iterations: int = 1
):
    return convolve_loop(image, u, v, kernel, iterations)
