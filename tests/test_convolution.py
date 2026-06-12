from dataclasses import dataclass
from itertools import combinations
from typing import Generic

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from numpy.typing import DTypeLike

import rlic
from rlic._typing import F, FArray1D, FArray2D

prng: np.random.Generator = np.random.default_rng(0)

NX: int = 128


@dataclass(kw_only=True, slots=True, frozen=True)
class ConvolveArgs(Generic[F]):
    img: FArray2D[F]
    u: FArray2D[F]
    v: FArray2D[F]
    kernel: FArray1D[F]


def get_convolve_args(
    nx: int = NX,
    klen: int = 11,
    dtype: np.dtype[F] = np.dtype("float64"),  # noqa: B008
) -> ConvolveArgs[F]:
    return ConvolveArgs(
        img=prng.random((nx, nx), dtype=dtype),
        u=prng.random((nx, nx), dtype=dtype),
        v=prng.random((nx, nx), dtype=dtype),
        kernel=np.linspace(0, 1, klen, dtype=dtype),
    )


ARGS: ConvolveArgs = get_convolve_args()


def test_no_iterations() -> None:
    out = rlic.convolve(ARGS.img, ARGS.u, ARGS.v, kernel=ARGS.kernel, iterations=0)
    assert_array_equal(out, ARGS.img)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_single_iteration(dtype: DTypeLike) -> None:
    args = get_convolve_args(dtype=dtype)
    out_impl = rlic.convolve(
        args.img,
        args.u,
        args.v,
        kernel=args.kernel,
    )
    out_expl = rlic.convolve(args.img, args.u, args.v, kernel=args.kernel, iterations=1)
    assert_array_equal(out_impl, out_expl)


def test_multiple_iterations() -> None:
    outs = [
        rlic.convolve(ARGS.img, ARGS.u, ARGS.v, kernel=ARGS.kernel, iterations=n)
        for n in range(3)
    ]
    for o1, o2 in combinations(outs, 2):
        assert np.all(o2 != o1)


def test_uv_symmetry() -> None:
    out1 = rlic.convolve(
        ARGS.img,
        ARGS.u,
        ARGS.v,
        kernel=ARGS.kernel,
    )
    out2 = rlic.convolve(
        ARGS.img.T,
        ARGS.v.T,
        ARGS.u.T,
        kernel=ARGS.kernel,
    ).T
    assert_array_equal(out2, out1)


def test_uv_mode_default() -> None:
    out_vel_impl = rlic.convolve(ARGS.img, ARGS.u, ARGS.v, kernel=ARGS.kernel)
    out_vel_expl = rlic.convolve(
        ARGS.img, ARGS.u, ARGS.v, kernel=ARGS.kernel, uv_mode="velocity"
    )
    assert_array_equal(out_vel_impl, out_vel_expl)


def test_uv_modes_diff() -> None:
    kernel = np.ones(5, dtype="float64")
    u0 = np.ones((NX, NX))
    ii = np.broadcast_to(np.arange(NX), (NX, NX))
    u1 = np.where(ii < NX / 2, u0, -u0)
    u2 = np.where(ii < NX / 2, -u0, u0)
    v = np.zeros((NX, NX))

    out_u1_vel = rlic.convolve(ARGS.img, u1, v, kernel=kernel, uv_mode="velocity")
    out_u2_vel = rlic.convolve(ARGS.img, u2, v, kernel=kernel, uv_mode="velocity")
    assert_allclose(out_u2_vel, out_u1_vel, atol=1e-14)

    out_u1_pol = rlic.convolve(ARGS.img, u1, v, kernel=kernel, uv_mode="polarization")
    out_u2_pol = rlic.convolve(ARGS.img, u2, v, kernel=kernel, uv_mode="polarization")
    assert_allclose(out_u2_pol, out_u1_pol, atol=1e-14)

    diff = out_u2_vel - out_u2_pol
    assert np.ptp(diff) > 1


@pytest.mark.parametrize("kernel_size", [3, 4])
def test_uv_modes_equiv(kernel_size: int) -> None:
    # with a kernel shorter than 5, uv_mode='polarization' doesn't do anything more or
    # different than uv_mode='velocity'
    kernel = np.ones(kernel_size, dtype="float64")
    out_vel = rlic.convolve(ARGS.img, ARGS.u, ARGS.v, kernel=kernel, uv_mode="velocity")
    out_pol = rlic.convolve(
        ARGS.img, ARGS.u, ARGS.v, kernel=kernel, uv_mode="polarization"
    )
    assert_array_equal(out_pol, out_vel)


def test_uv_mode_polarization_sym() -> None:
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
def test_nan_vectors(dtype: np.dtype[F], niterations: int) -> None:
    args = get_convolve_args(dtype=dtype)
    u = v = np.full_like(args.img, np.nan)

    # streamlines will all be terminated on the first step,
    # but the starting pixel is still to be accumulated, so we expect
    # the output to be identical to the input, to a scaling factor.
    out = rlic.convolve(args.img, u, v, kernel=args.kernel, iterations=niterations)
    scaling_factor = out / args.img
    assert np.ptp(scaling_factor) == 0.0
    assert scaling_factor[0, 0] == args.kernel[len(args.kernel) // 2] ** niterations


def test_boundaries() -> None:
    args = get_convolve_args(dtype=np.dtype("float64"), nx=64, klen=128)
    ONE = np.ones_like(args.img)
    ZERO = np.zeros_like(args.img)
    U0 = ONE
    nx, ny = args.img.shape
    x = np.linspace(0, np.pi, ny)
    ii = np.broadcast_to(np.arange(nx), args.img.shape)
    U = np.where(ii < nx / 2, -U0, U0)
    V = np.broadcast_to(np.sin(x).T, args.img.shape)
    out_closed = rlic.convolve(args.img, U, V, kernel=args.kernel, boundaries="closed")
    out_period = rlic.convolve(
        args.img, U, V, kernel=args.kernel, boundaries="periodic"
    )

    assert_array_less(ZERO, np.abs(out_closed - out_period))

    out12 = rlic.convolve(
        args.img, U, V, kernel=args.kernel, boundaries={"x": "closed", "y": "periodic"}
    )
    out21 = rlic.convolve(
        args.img, U, V, kernel=args.kernel, boundaries={"y": "closed", "x": "periodic"}
    )
    # assert_array_less(ZERO, np.abs(out12 - out_closed))
    assert_array_less(ZERO, np.abs(out12 - out_period))
    assert_array_less(ZERO, np.abs(out21 - out_closed))
    # assert_array_less(ZERO, np.abs(out21 - out_period))
    assert_array_less(ZERO, np.abs(out12 - out21))
