import numpy as np
import rlic
from numpy.testing import assert_array_equal
import pytest

img = np.eye(5)
kernel = np.linspace(0,1,10)

def test_invalid_iterations():
    with pytest.raises(ValueError):
        rlic.convolve(img, img, img, kernel=kernel, iterations=-1)

def test_no_iterations():
    out = rlic.convolve(img, img, img, kernel=kernel, iterations=0)
    assert_array_equal(out, img)

def test_single_iteration():
    out = rlic.convolve(img, img, img, kernel=kernel, iterations=1)
    # TODO: add some assertions