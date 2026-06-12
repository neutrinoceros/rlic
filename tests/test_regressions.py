from itertools import product

import numpy as np
import pytest
from _pytest.fixtures import SubRequest
from numpy.testing import assert_allclose

import rlic

pytest.importorskip("vectorplot")
from dataclasses import dataclass

from vectorplot.lic_internal import (  # noqa: E402
    line_integral_convolution as reference_impl,
)

from rlic._typing import D2, FArray1D, FArray2D, UVMode

SHAPE: tuple[int, int] = (31, 33)
NX: int = SHAPE[0]
NY: int = SHAPE[1]
rng: np.random.Generator = np.random.default_rng(0)
TEXTURE: FArray2D = rng.random(SHAPE, dtype="float32")

ONE: FArray2D = np.ones_like(TEXTURE)
ZERO: FArray2D = np.zeros_like(TEXTURE)

# define velocity components with a sharp divergence
ii: np.ndarray[D2, np.dtype[np.int64]] = np.broadcast_to(np.arange(NY), SHAPE)
cond: np.ndarray[D2, np.dtype[np.bool_]] = ii < NY / 2
U1: FArray2D = np.where(cond, -ONE, ONE)

jj: np.ndarray[D2, np.dtype[np.int64]] = np.broadcast_to(
    np.atleast_2d(np.arange(NX)).T, SHAPE
)
cond = jj < NX / 2
V1: FArray2D = np.where(cond, -ONE, ONE)

KL: int = min(NX, NY)
K0: FArray1D = np.sin(np.arange(KL) * np.pi / KL, dtype="float32")


@dataclass(kw_only=True, slots=True, frozen=True)
class Case:
    u: FArray2D
    v: FArray2D
    uv_mode: UVMode

    def expected_result(self) -> FArray2D:
        return reference_impl(
            self.u, self.v, TEXTURE, K0, int(self.uv_mode == "polarization")
        )


TEST_CASES: list[Case] = [
    Case(u=u, v=v, uv_mode=uv_mode)
    for (u, v, uv_mode) in product([ZERO, U1], [ZERO, V1], ["velocity", "polarization"])
]


@pytest.fixture(params=TEST_CASES)
def case(request: SubRequest) -> Case:
    return request.param


def test_outputs(case: Case) -> None:
    out = rlic.convolve(TEXTURE, case.u, case.v, kernel=K0, uv_mode=case.uv_mode)
    assert_allclose(out, case.expected_result(), rtol=1.5e-7, atol=1e-6)
