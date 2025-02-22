import numpy as np
import rlic
import pytest

img = np.eye(5)
kernel = np.linspace(0,1,10)

def test_invalid_iterations():
    with pytest.raises(ValueError):
        rlic.convolve(img, img, img, kernel=kernel, iterations=-1)
