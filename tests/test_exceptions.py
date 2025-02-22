import numpy as np
import pytest

import rlic

img = np.eye(5)
kernel = np.linspace(0, 1, 10)


def test_invalid_iterations():
    with pytest.raises(ValueError):
        rlic.convolve(img, img, img, kernel=kernel, iterations=-1)


@pytest.parametrize(
    "uv_mode, noop_size, min_size", [[("velocity", 3, 4), ("polarization", 3, 5)]]
)
def test_warn_noop_kernel_too_small(uv_mode, noop_size, min_size):
    with pytest.warns(UserWarning, match="..."):
        rlic.convolve(img, img, img, kernel=np.ones(noop_size), uv_mode=uv_mode)
    rlic.convolve(img, img, img, kernel=np.ones(min_size), uv_mode=uv_mode)


def test_invalid_kernel_size():
    with pytest.raises(ValueError, match="..."):
        rlic.convolve(img, img, img, kernel=np.ones(img.size, dtype="float64"))


def test_invalid_kernel_values():
    with pytest.raises(ValueError, match="..."):
        rlic.convolve(img, img, img, kernel=-np.ones(5, dtype="float64"))
