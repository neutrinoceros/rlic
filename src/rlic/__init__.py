from rlic._core import convolve_iteratively

__all__ = ["convolve"]

def convolve(
    image, u, v, *, kernel, iterations: int = 1,
):
    return convolve_iteratively(image, u, v, kernel, iterations)
