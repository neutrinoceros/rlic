import numpy as np
import pytest

import rlic

img = u = v = np.eye(64)
kernel = np.linspace(0, 1, 10)


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


def test_invalid_image_ndim():
    img = np.ones((16, 16, 16))
    with pytest.raises(
        ValueError,
        match=r"^Expected an image with exactly two dimensions\. Got image\.ndim=3$",
    ):
        rlic.convolve(img, u, v, kernel=kernel)


def test_invalid_image_values():
    img = -np.ones((64, 64))
    with pytest.raises(
        ValueError,
        match=(r"^Found invalid image element\(s\)\. Expected only positive values\.$"),
    ):
        rlic.convolve(img, v, v, kernel=kernel)


@pytest.mark.parametrize(
    "image_shape, u_shape, v_shape",
    [
        ((64, 64), (65, 64), (64, 64)),
        ((64, 64), (64, 64), (63, 64)),
        ((64, 66), (64, 64), (64, 64)),
    ],
)
def test_mismatched_shapes(image_shape, u_shape, v_shape):
    prng = np.random.default_rng(0)
    image = prng.random(image_shape)
    u = prng.random(u_shape)
    v = prng.random(v_shape)
    with pytest.raises(
        ValueError,
        match=(
            r"^Shape mismatch: expected image, u and v with identical shapes\. "
            rf"Got image.shape=\({image.shape[0]}, {image.shape[1]}\), "
            rf"u.shape=\({u.shape[0]}, {u.shape[1]}\), "
            rf"v.shape=\({v.shape[0]}, {v.shape[1]}\)$"
        ),
    ):
        rlic.convolve(image, u, v, kernel=kernel)


def test_invalid_kernel_ndim():
    with pytest.raises(
        ValueError,
        match=r"^Expected a kernel with exactly one dimension\. Got kernel\.ndim=2$",
    ):
        rlic.convolve(img, u, v, kernel=np.ones((5, 5)))


def test_kernel_too_small():
    with pytest.raises(
        ValueError,
        match=r"^Expected a kernel with size 3 or more\. Got kernel\.size=2$",
    ):
        rlic.convolve(img, u, v, kernel=np.ones(2))
    rlic.convolve(img, u, v, kernel=np.ones(3))


def test_kernel_too_long():
    with pytest.raises(
        ValueError,
        match=rf"^kernel\.size={img.size} exceeds the smallest dim of the image \({len(img)}\)$",
    ):
        rlic.convolve(img, u, v, kernel=np.ones(img.size, dtype="float64"))


def test_invalid_kernel_values():
    with pytest.raises(
        ValueError,
        match=r"^Found invalid kernel element\(s\)\. Expected only positive values\.$",
    ):
        rlic.convolve(img, u, v, kernel=-np.ones(5, dtype="float64"))


def test_invalid_image_dtype():
    img = np.ones((64, 64), dtype="complex128")
    with pytest.raises(
        TypeError,
        match=(
            r"^Found unsupported data type\(s\): \[dtype\('complex128'\)\]\. "
            r"Expected image, u, v and kernel with identical dtype, from "
            r"\[dtype\('float32'\), dtype\('float64'\)\]\. "
            r"Got image\.dtype=dtype\('complex128'\), u\.dtype=dtype\('float64'\), "
            r"v\.dtype=dtype\('float64'\), kernel\.dtype=dtype\('float64'\)$"
        ),
    ):
        rlic.convolve(img, u, v, kernel=kernel)


def test_invalid_kernel_dtype():
    with pytest.raises(
        TypeError,
        match=(
            r"^Found unsupported data type\(s\): \[dtype\('complex128'\)\]\. "
            r"Expected image, u, v and kernel with identical dtype, from "
            r"\[dtype\('float32'\), dtype\('float64'\)\]\. "
            r"Got image\.dtype=dtype\('float64'\), u\.dtype=dtype\('float64'\), "
            r"v\.dtype=dtype\('float64'\), kernel\.dtype=dtype\('complex128'\)$"
        ),
    ):
        rlic.convolve(img, u, v, kernel=-np.ones(5, dtype="complex128"))
