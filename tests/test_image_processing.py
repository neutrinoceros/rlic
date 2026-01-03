import numpy as np
import numpy.testing as npt
import pytest

import rlic
from rlic._histeq_adaptive_strategy import Strategy


def test_missing_strategy_kind():
    with pytest.raises(TypeError, match=r"^strategy dict is missing a 'kind' key\.$"):
        Strategy.from_spec({})


def test_unknown_strategy_kind():
    with pytest.raises(
        ValueError,
        match=(
            r"^Unknown strategy kind 'not-a-kind'\. "
            r"Expected one of \['sliding-window'\]$"
        ),
    ):
        Strategy.from_spec({"kind": "not-a-kind"})


def test_sliding_window_missing_tile_size():
    with pytest.raises(
        TypeError,
        match=(
            r"Neither 'tile-size' nor 'tile-size-max' keys were found\. "
            r"Either are allowed, but exactly one is expected\."
        ),
    ):
        Strategy.from_spec({"kind": "sliding-window"})


def test_sliding_window_both_tile_sizes_keys():
    with pytest.raises(
        TypeError,
        match=(
            r"Both 'tile-size' and 'tile-size-max' keys were provided\. "
            r"Only one of them can be specified at a time\."
        ),
    ):
        Strategy.from_spec(
            {"kind": "sliding-window", "tile-size": 11, "tile-size-max": 13}
        )


@pytest.mark.parametrize(
    "spec, expected",
    [
        pytest.param(
            {"kind": "sliding-window", "tile-size": 13},
            Strategy(kind="sliding-window", tile_size=(13, 13)),
            id="int-tile-size",
        ),
        pytest.param(
            {"kind": "sliding-window", "tile-size": (13, 15)},
            Strategy(kind="sliding-window", tile_size=(13, 15)),
            id="tuple-tile-size",
        ),
        pytest.param(
            {"kind": "sliding-window", "tile-size-max": 13},
            Strategy(kind="sliding-window", tile_size_max=(13, 13)),
            id="int-tile-size-max",
        ),
        pytest.param(
            {"kind": "sliding-window", "tile-size-max": (13, 15)},
            Strategy(kind="sliding-window", tile_size_max=(13, 15)),
            id="tuple-tile-size-max",
        ),
    ],
)
def test_sliding_window_from_spec(spec, expected):
    strategy = Strategy.from_spec(spec)
    assert strategy == expected


def _equalize_histogram_numpy(image, nbins):
    # ref implementation, adapted from scikit-image (exposure.equalize_hist)
    flat_image = image.ravel()

    hist, bin_edges = np.histogram(flat_image, bins=nbins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # cumulative distribution function
    cdf = hist.cumsum(dtype=image.dtype)
    normalized_cdf = cdf / cdf[-1]

    # As of version 2.4, np.interp always promotes to float64, so we
    # have to cast back to single precision when float32 output is desired
    return (
        np.interp(image.flat, bin_centers, normalized_cdf)
        .reshape(image.shape)
        .astype(image.dtype, copy=False)
    )


@pytest.mark.parametrize(
    "nbins, min_rms_reduction",
    [
        (12, 2.0),
        (64, 10.0),
        (256, 50.0),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_historgram_equalization(nbins, min_rms_reduction, dtype, subtests):
    # histogram equalization produces a new image whose cumulative
    # distribution function (cdf) should be close(r) to a straight line
    # (i.e., approaching a flat intensity distribution)
    # This test check this property from an initial image made of gaussian
    # noise.
    # Expected rms reduction factors (min_rms_reduction) are empirical, i.e.,
    # slightly looser than what the original implementation was able to achieve

    IMAGE_SHAPE = (256, 128)
    prng = np.random.default_rng(0)
    image = np.clip(
        prng.normal(loc=5.0, scale=1.0, size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE),
        a_min=0.0,
        a_max=None,
        dtype=dtype,
    )

    def normalized_cdf(a):
        hist, bin_edges = np.histogram(a.ravel(), bins=nbins)
        cdf = hist.cumsum(dtype=dtype)
        return cdf / cdf[-1]

    def rms(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    image_eq = rlic.equalize_histogram(image, nbins=nbins)
    cdf_in = normalized_cdf(image)
    cdf_eq = normalized_cdf(image_eq)

    id_func = np.linspace(0, 1, nbins)
    rms_in = rms(cdf_in, id_func)
    rms_eq = rms(cdf_eq, id_func)
    with subtests.test("basic properties"):
        assert rms_eq < rms_in
        assert (rms_in / rms_eq) > min_rms_reduction

    image_ref = _equalize_histogram_numpy(image, nbins=nbins)

    match dtype:
        case "float32":
            rtol = 2e-6
        case "float64":
            rtol = 2e-15
        case _ as _unreachable:
            raise AssertionError
    with subtests.test("compare to ref impl"):
        npt.assert_allclose(image_eq, image_ref, rtol=rtol)


def test_historgram_equalization_unsupported_dtype():
    with pytest.raises(
        TypeError,
        match=(
            r"^Found unsupported data type: int64\. "
            r"Expected of of \[dtype\('float32'\), dtype\('float64'\)\]\.$"
        ),
    ):
        rlic.equalize_histogram(np.eye(3, dtype="int64"), nbins=3)
