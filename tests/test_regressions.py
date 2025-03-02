from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

import rlic

pytest.importorskip("vectorplot")
from vectorplot.lic_internal import (  # noqa: E402
    line_integral_convolution as reference_impl,
)

NX, NY = SHAPE = (31, 33)
rng = np.random.default_rng(0)
texture = rng.random(SHAPE, dtype="float32")

ONE = np.ones_like(texture)
ZERO = np.zeros_like(texture)

# define velocity components with a sharp divergence
ii = np.broadcast_to(np.arange(NY), SHAPE)
cond = ii < NY / 2
U1 = np.where(cond, -ONE, ONE)

jj = np.broadcast_to(np.atleast_2d(np.arange(NX)).T, SHAPE)
cond = jj < NX / 2
V1 = np.where(cond, -ONE, ONE)

KL = min(NX, NY)
K0 = np.sin(np.arange(KL) * np.pi / KL, dtype="float32")

test_cases = sorted(
    (u, v, k, uv_mode)
    for u, v, k, uv_mode in product(["u0", "u1"], ["v0", "v1"], ["k0"], ["vel", "pol"])
)


@pytest.fixture(params=test_cases)
def known_answer(request):
    field_u, field_v, field_kernel, field_mode = request.param
    if field_u == "u0":
        u = ZERO
    elif field_u == "u1":
        u = U1
    else:
        raise RuntimeError  # pragma: no cover

    if field_v == "v0":
        v = ZERO
    elif field_v == "v1":
        v = V1
    else:
        raise RuntimeError  # pragma: no cover

    if field_kernel == "k0":
        kernel = K0
    else:
        raise RuntimeError  # pragma: no cover

    if field_mode == "vel":
        uv_mode = "velocity"
    elif field_mode == "pol":
        uv_mode = "polarization"
    else:
        raise RuntimeError  # pragma: no cover

    return (
        u,
        v,
        kernel,
        uv_mode,
        reference_impl(u, v, texture, kernel, int(uv_mode == "polarization")),
    )


def test_outputs(known_answer):
    u, v, kernel, uv_mode, expected = known_answer
    out = rlic.convolve(texture, u, v, kernel=kernel, uv_mode=uv_mode)
    assert_allclose(out, expected, rtol=1.5e-7, atol=1e-6)
