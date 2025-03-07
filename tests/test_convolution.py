from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import rlic

prng = np.random.default_rng(0)

NX = 128


def get_convolve_args(nx=NX, dtype="float64"):
    return (
        prng.random((nx, nx), dtype=dtype),
        prng.random((nx, nx), dtype=dtype),
        prng.random((nx, nx), dtype=dtype),
        np.linspace(0, 1, 11, dtype=dtype),
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


def test_uv_mode_default():
    out_vel_impl = rlic.convolve(img, u, v, kernel=kernel)
    out_vel_expl = rlic.convolve(img, u, v, kernel=kernel, uv_mode="velocity")
    assert_array_equal(out_vel_impl, out_vel_expl)


def test_uv_modes_diff():
    kernel = np.ones(5, dtype="float64")
    u0 = np.ones((NX, NX))
    ii = np.broadcast_to(np.arange(NX), (NX, NX))
    u1 = np.where(ii < NX / 2, u0, -u0)
    u2 = np.where(ii < NX / 2, -u0, u0)
    v = np.zeros((NX, NX))

    out_u1_vel = rlic.convolve(img, u1, v, kernel=kernel, uv_mode="velocity")
    out_u2_vel = rlic.convolve(img, u2, v, kernel=kernel, uv_mode="velocity")
    assert_allclose(out_u2_vel, out_u1_vel, atol=1e-14)

    out_u1_pol = rlic.convolve(img, u1, v, kernel=kernel, uv_mode="polarization")
    out_u2_pol = rlic.convolve(img, u2, v, kernel=kernel, uv_mode="polarization")
    assert_allclose(out_u2_pol, out_u1_pol, atol=1e-14)

    diff = out_u2_vel - out_u2_pol
    assert np.ptp(diff) > 1


@pytest.mark.parametrize("kernel_size", [3, 4])
def test_uv_modes_equiv(kernel_size):
    # with a kernel shorter than 5, uv_mode='polarization' doesn't do anything more or
    # different than uv_mode='velocity'
    kernel = np.ones(kernel_size, dtype="float64")
    out_vel = rlic.convolve(img, u, v, kernel=kernel, uv_mode="velocity")
    out_pol = rlic.convolve(img, u, v, kernel=kernel, uv_mode="polarization")
    assert_array_equal(out_pol, out_vel)


def test_uv_mode_polarization_sym():
    NX = 5
    kernel = np.array([1, 1, 1, 1, 1], dtype="float64")
    shape = (NX, NX)
    img = np.eye(NX)
    ZERO = np.zeros(shape, dtype="float64")
    ONE = np.ones(shape, dtype="float64")
    out_u_forward = rlic.convolve(
        img,
        u=ONE,
        v=ZERO,
        kernel=kernel,
        uv_mode="polarization",
    )
    out_u_backward = rlic.convolve(
        img,
        u=-ONE,
        v=ZERO,
        kernel=kernel,
        uv_mode="polarization",
    )
    assert_allclose(out_u_backward, out_u_forward)

    out_v_forward = rlic.convolve(
        img,
        u=ZERO,
        v=ONE,
        kernel=kernel,
        uv_mode="polarization",
    )
    out_v_backward = rlic.convolve(
        img,
        u=ZERO,
        v=-ONE,
        kernel=kernel,
        uv_mode="polarization",
    )
    assert_allclose(out_v_backward, out_v_forward)


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
