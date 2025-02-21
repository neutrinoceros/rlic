import numpy as np
import rlic
from numpy.testing import assert_array_equal

img = np.eye(5)
kernel = np.linspace(0,1,10)

def test_no_iterations():
    out = rlic.convolve(img, img, img, kernel=kernel, iterations=0)
    assert_array_equal(out, img)
