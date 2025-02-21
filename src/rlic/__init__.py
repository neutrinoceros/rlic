from rlic._core import convolve_iteratively

__all__ = ["convolve"]

def convolve(
    image, u, v, *, kernel, iterations: int = 1,
):
    if iterations < 0:
        raise ValueError(f"Invalid number of iterations {iterations}. "
                         "Expected a strictly positive integer.")
    if iterations == 0:
        return image.copy()
    return convolve_iteratively(image, u, v, kernel, iterations)
