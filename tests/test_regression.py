import os
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import rlic

NX, NY = SHAPE = (31, 33)
rng = np.random.default_rng(0)
texture = rng.random(SHAPE, dtype="float64")

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
K0 = np.sin(np.arange(KL) * np.pi / KL)
DATA_DIR = Path(__file__).parent / "data" / "regressions"

test_cases = sorted(
    f"{u}_{v}_{k}_{uv_mode}"
    for u, v, k, uv_mode in product(["u0", "u1"], ["v0", "v1"], ["k0"], ["vel", "pol"])
)


@pytest.fixture(params=test_cases)
def known_answer(request):
    data_file = DATA_DIR.joinpath(request.param).with_suffix(".npy")
    field_u, field_v, field_kernel, field_mode = data_file.stem.split("_")
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

    if os.getenv("RLIC_GENERATE_ANSWERS", "0") == "1":  # pragma: no cover
        reference_output = rlic.convolve(texture, u, v, kernel=kernel, uv_mode=uv_mode)
        np.save(data_file, reference_output)
    elif data_file.exists():
        reference_output = np.load(data_file)
    else:
        raise RuntimeError  # pragma: no cover
    return u, v, kernel, uv_mode, reference_output


def test_outputs(known_answer):
    u, v, kernel, uv_mode, expected = known_answer
    out = rlic.convolve(texture, u, v, kernel=kernel, uv_mode=uv_mode)
    assert_allclose(out, expected, rtol=5e-16, atol=5e-16)
