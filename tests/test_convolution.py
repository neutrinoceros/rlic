from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_less

import rlic

prng = np.random.default_rng(0)

NX = 128


def get_convolve_args(nx=NX, klen=11, dtype="float64"):
    return (
        prng.random((nx, nx), dtype=dtype),
        prng.random((nx, nx), dtype=dtype),
        prng.random((nx, nx), dtype=dtype),
        np.linspace(0, 1, klen, dtype=dtype),
    )


img, u, v, kernel = get_convolve_args()


def test_no_iterations():
    out = rlic.convolve(img, u, v, kernel=kernel, iterations=0)
    assert_array_equal(out, img)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_single_iteration(dtype):
    img, u, v, kernel = get_convolve_args(dtype=dtype)
    out_impl = rlic.convolve(img, u, v, kernel=kernel)
    out_expl = rlic.convolve(img, u, v, kernel=kernel, iterations=1)
    assert_array_equal(out_impl, out_expl)


def test_multiple_iterations():
    outs = [rlic.convolve(img, u, v, kernel=kernel, iterations=n) for n in range(3)]
    for o1, o2 in combinations(outs, 2):
        assert np.all(o2 != o1)


def test_uv_symmetry():
    out1 = rlic.convolve(img, u, v, kernel=kernel)
    out2 = rlic.convolve(img.T, v.T, u.T, kernel=kernel).T
    assert_array_equal(out2, out1)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("niterations", [0, 1, 5])
def test_nan_vectors(dtype, niterations):
    img, _, _, kernel = get_convolve_args(dtype=dtype)
    u = v = np.full_like(img, np.nan)

    # streamlines will all be terminated on the first step,
    # but the starting pixel is still to be accumulated, so we expect
    # the output to be identical to the input, to a scaling factor.
    out = rlic.convolve(img, u, v, kernel=kernel, iterations=niterations)
    scaling_factor = out / img
    assert np.ptp(scaling_factor) == 0.0
    assert scaling_factor[0, 0] == kernel[len(kernel) // 2] ** niterations


def test_boundaries():
    img, _, _, kernel = get_convolve_args(dtype="float64", nx=64, klen=128)
    ONE = np.ones_like(img)
    ZERO = np.zeros_like(img)
    U0 = ONE
    nx, ny = img.shape
    x = np.linspace(0, np.pi, ny)
    ii = np.broadcast_to(np.arange(nx), img.shape)
    U = np.where(ii < nx / 2, -U0, U0)
    V = np.broadcast_to(np.sin(x).T, img.shape)
    out_closed = rlic.convolve(img, U, V, kernel=kernel, boundaries="closed")
    out_period = rlic.convolve(img, U, V, kernel=kernel, boundaries="periodic")

    assert_array_less(ZERO, np.abs(out_closed - out_period))

    out12 = rlic.convolve(
        img, U, V, kernel=kernel, boundaries={"x": "closed", "y": "periodic"}
    )
    out21 = rlic.convolve(
        img, U, V, kernel=kernel, boundaries={"y": "closed", "x": "periodic"}
    )
    # assert_array_less(ZERO, np.abs(out12 - out_closed))
    assert_array_less(ZERO, np.abs(out12 - out_period))
    assert_array_less(ZERO, np.abs(out21 - out_closed))
    # assert_array_less(ZERO, np.abs(out21 - out_period))
    assert_array_less(ZERO, np.abs(out12 - out21))
