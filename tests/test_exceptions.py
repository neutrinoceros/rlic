import numpy as np
import pytest

import rlic

img = np.eye(64)
kernel = np.linspace(0, 1, 10)


def test_invalid_iterations():
    with pytest.raises(
        ValueError,
        match=(
            r"^Invalid number of iterations: -1\n"
            r"Expected a strictly positive integer\.$"
        ),
    ):
        rlic.convolve(img, img, img, kernel=kernel, iterations=-1)


def test_kernel_too_small():
    with pytest.raises(
        ValueError,
        match=r"^Expected a kernel with size 3 or more\. Got kernel\.size=2$",
    ):
        rlic.convolve(img, img, img, kernel=np.ones(2))
    rlic.convolve(img, img, img, kernel=np.ones(3))


def test_kernel_too_long():
    with pytest.raises(
        ValueError,
        match=rf"^kernel\.size={img.size} exceeds the smallest dim of the image \({len(img)}\)$",
    ):
        rlic.convolve(img, img, img, kernel=np.ones(img.size, dtype="float64"))


def test_invalid_kernel_values():
    with pytest.raises(
        ValueError,
        match=r"^Found invalid kernel element\(s\)\. Expected only positive values\.$",
    ):
        rlic.convolve(img, img, img, kernel=-np.ones(5, dtype="float64"))
