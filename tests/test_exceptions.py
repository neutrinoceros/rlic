import sys

import numpy as np
import pytest

import rlic

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

img = u = v = np.eye(64)
kernel = np.linspace(0, 1, 10, dtype="float64")


def assert_exceptions_match(
    to_eval: str,
    global_namespace,
    local_namespace,
    group_match: str,
    expected_sub: list[tuple[type[Exception], str]],
) -> None:
    with pytest.raises(ExceptionGroup, match=group_match) as excinfo:
        eval(to_eval, global_namespace, local_namespace)

    for exctype, match in expected_sub:
        assert excinfo.group_contains(exctype, match=match, depth=1)

    assert len(excinfo.value.exceptions) == len(expected_sub)


def test_invalid_iterations():
    with pytest.raises(
        ValueError,
        match=(
            r"^Invalid number of iterations: -1\n"
            r"Expected a strictly positive integer\.$"
        ),
    ):
        rlic.convolve(img, u, v, kernel=kernel, iterations=-1)


def test_invalid_uv_mode():
    with pytest.raises(
        ValueError,
        match=(
            r"^Invalid uv_mode 'astral'\. Expected one of \['velocity', 'polarization'\]$"
        ),
    ):
        rlic.convolve(img, u, v, kernel=kernel, uv_mode="astral")


def test_invalid_texture_ndim():
    img = np.ones((16, 16, 16))
    assert_exceptions_match(
        "rlic.convolve(img, u, v, kernel=kernel)",
        globals(),
        locals(),
        group_match=r"^Invalid inputs were received\.",
        expected_sub=[
            (
                ValueError,
                r"^Expected a texture with exactly two dimensions\. Got texture\.ndim=3$",
            ),
            (
                ValueError,
                r"^Shape mismatch: expected texture, u and v with identical shapes\.",
            ),
        ],
    )


def test_invalid_texture_shape_and_ndim():
    img = np.ones((16, 16, 16))

    assert_exceptions_match(
        "rlic.convolve(img, u, v, kernel=kernel)",
        globals(),
        locals(),
        group_match=r"^Invalid inputs were received\.",
        expected_sub=[
            (
                ValueError,
                r"^Expected a texture with exactly two dimensions\. Got texture\.ndim=3$",
            ),
            (
                ValueError,
                r"^Shape mismatch: expected texture, u and v with identical shapes\.",
            ),
        ],
    )


def test_invalid_texture_values():
    img = -np.ones((64, 64))
    with pytest.raises(
        ValueError,
        match=(
            r"^Found invalid texture element\(s\)\. Expected only positive values\.$"
        ),
    ):
        rlic.convolve(img, v, v, kernel=kernel)


@pytest.mark.parametrize(
    "texture_shape, u_shape, v_shape",
    [
        ((64, 64), (65, 64), (64, 64)),
        ((64, 64), (64, 64), (63, 64)),
        ((64, 66), (64, 64), (64, 64)),
    ],
)
def test_mismatched_shapes(texture_shape, u_shape, v_shape):
    prng = np.random.default_rng(0)
    texture = prng.random(texture_shape)
    u = prng.random(u_shape)
    v = prng.random(v_shape)
    with pytest.raises(
        ValueError,
        match=(
            r"^Shape mismatch: expected texture, u and v with identical shapes\. "
            rf"Got texture.shape=\({texture.shape[0]}, {texture.shape[1]}\), "
            rf"u.shape=\({u.shape[0]}, {u.shape[1]}\), "
            rf"v.shape=\({v.shape[0]}, {v.shape[1]}\)$"
        ),
    ):
        rlic.convolve(texture, u, v, kernel=kernel)


def test_invalid_kernel_ndim():
    with pytest.raises(
        ValueError,
        match=r"^Expected a kernel with exactly one dimension\. Got kernel\.ndim=2$",
    ):
        rlic.convolve(img, u, v, kernel=np.ones((5, 5)))


@pytest.mark.parametrize("polluting_value", [-np.inf, np.inf, np.nan])
def test_non_finite_kernel(polluting_value):
    kernel = np.ones(11)
    kernel[5] = polluting_value
    with pytest.raises(
        ValueError,
        match=r"^Found non-finite value\(s\) in kernel\.$",
    ):
        rlic.convolve(img, u, v, kernel=kernel)


def test_invalid_texture_dtype():
    img = np.ones((64, 64), dtype="complex128")
    assert_exceptions_match(
        "rlic.convolve(img, u, v, kernel=kernel)",
        globals(),
        locals(),
        group_match=r"^Invalid inputs were received\.",
        expected_sub=[
            (
                TypeError,
                r"^Found unsupported data type\(s\): \[dtype\('complex128'\)\]\. "
                r"Expected texture, u, v and kernel with identical dtype, from "
                r"\[dtype\('float32'\), dtype\('float64'\)\]\. "
                r"Got texture\.dtype=dtype\('complex128'\), u\.dtype=dtype\('float64'\), "
                r"v\.dtype=dtype\('float64'\), kernel\.dtype=dtype\('float64'\)$",
            ),
            (TypeError, r"^Data types mismatch"),
        ],
    )


def test_invalid_kernel_dtype():
    assert_exceptions_match(
        "rlic.convolve(img, u, v, kernel=-np.ones(5, dtype='complex128'))",
        globals(),
        locals(),
        group_match=r"^Invalid inputs were received\.",
        expected_sub=[
            (
                TypeError,
                r"^Found unsupported data type\(s\): \[dtype\('complex128'\)\]\. "
                r"Expected texture, u, v and kernel with identical dtype, from "
                r"\[dtype\('float32'\), dtype\('float64'\)\]\. "
                r"Got texture\.dtype=dtype\('float64'\), u\.dtype=dtype\('float64'\), "
                r"v\.dtype=dtype\('float64'\), kernel\.dtype=dtype\('complex128'\)$",
            ),
            (TypeError, r"^Data types mismatch"),
        ],
    )


def test_mismatched_dtypes():
    img = np.ones((64, 64), dtype="float32")
    with pytest.raises(
        TypeError,
        match=(
            r"^Data types mismatch. "
            r"Expected texture, u, v and kernel with identical dtype, from "
            r"\[dtype\('float32'\), dtype\('float64'\)\]\. "
            r"Got texture\.dtype=dtype\('float32'\), u\.dtype=dtype\('float64'\), "
            r"v\.dtype=dtype\('float64'\), kernel\.dtype=dtype\('float64'\)$"
        ),
    ):
        rlic.convolve(img, u, v, kernel=kernel)


def test_all_validators_before_returns():
    # until v0.3.2, iterations=0 implied an early return that skipped
    # most validators.
    kernel = np.full(11, np.nan)
    with pytest.raises(
        ValueError,
        match=r"^Found non-finite value\(s\) in kernel\.$",
    ):
        rlic.convolve(img, u, v, kernel=kernel, iterations=0)
