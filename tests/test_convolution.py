from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import rLIC

prng = np.random.default_rng(0)

NX = 128
img = prng.random((NX, NX))
u = prng.random((NX, NX))
v = prng.random((NX, NX))
kernel = np.linspace(0, 1, 10)


def test_no_iterations():
    out = rLIC.convolve(img, u, v, kernel=kernel, iterations=0)
    assert_array_equal(out, img)


def test_single_iteration():
    out_impl = rLIC.convolve(img, u, v, kernel=kernel)
    out_expl = rLIC.convolve(img, u, v, kernel=kernel, iterations=1)
    assert_array_equal(out_impl, out_expl)


def test_multiple_iterations():
    outs = [rLIC.convolve(img, u, v, kernel=kernel, iterations=n) for n in range(3)]
    for o1, o2 in combinations(outs, 2):
        assert np.all(o2 != o1)


def test_uv_mode_default():
    out_vel_impl = rLIC.convolve(img, u, v, kernel=kernel)
    out_vel_expl = rLIC.convolve(img, u, v, kernel=kernel, uv_mode="velocity")
    assert_array_equal(out_vel_impl, out_vel_expl)


def test_uv_modes_diff():
    kernel = np.ones(5, dtype="float64")
    u0 = np.ones((NX, NX))
    ii = np.broadcast_to(np.arange(NX), (NX, NX))
    u1 = np.where(ii < NX / 2, u0, -u0)
    u2 = np.where(ii < NX / 2, -u0, u0)
    v = np.zeros((NX, NX))

    out_u1_vel = rLIC.convolve(img, u1, v, kernel=kernel, uv_mode="velocity")
    out_u2_vel = rLIC.convolve(img, u2, v, kernel=kernel, uv_mode="velocity")
    assert_allclose(out_u2_vel, out_u1_vel, atol=1e-14)

    out_u1_pol = rLIC.convolve(img, u1, v, kernel=kernel, uv_mode="polarization")
    out_u2_pol = rLIC.convolve(img, u2, v, kernel=kernel, uv_mode="polarization")
    assert_allclose(out_u2_pol, out_u1_pol, atol=1e-14)

    diff = out_u2_vel - out_u2_pol
    assert np.ptp(diff) > 1


@pytest.mark.parametrize("kernel_size", [3, 4])
def test_uv_modes_equiv(kernel_size):
    # with a kernel shorter than 5, uv_mode='polarization' doesn't do anything more or
    # different than uv_mode='velocity'
    kernel = np.ones(kernel_size, dtype="float64")
    out_vel = rLIC.convolve(img, u, v, kernel=kernel, uv_mode="velocity")
    out_pol = rLIC.convolve(img, u, v, kernel=kernel, uv_mode="polarization")
    assert_array_equal(out_pol, out_vel)


# TODO:
# - test that with kernel size < 5, both uv_mode options are equivalent
# - document it


def test_uv_mode_polarization_sym():
    NX = 5
    kernel = np.array([1, 1, 1, 1, 1], dtype="float64")
    shape = (NX, NX)
    img = np.eye(NX)
    ZERO = np.zeros(shape, dtype="float64")
    ONE = np.ones(shape, dtype="float64")
    out_u_forward = rLIC.convolve(
        img,
        u=ONE,
        v=ZERO,
        kernel=kernel,
        uv_mode="polarization",
    )
    out_u_backward = rLIC.convolve(
        img,
        u=-ONE,
        v=ZERO,
        kernel=kernel,
        uv_mode="polarization",
    )
    assert_allclose(out_u_backward, out_u_forward)

    out_v_forward = rLIC.convolve(
        img,
        u=ZERO,
        v=ONE,
        kernel=kernel,
        uv_mode="polarization",
    )
    out_v_backward = rLIC.convolve(
        img,
        u=ZERO,
        v=-ONE,
        kernel=kernel,
        uv_mode="polarization",
    )
    assert_allclose(out_v_backward, out_v_forward)
