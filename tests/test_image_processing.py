import re
import sys

import numpy as np
import numpy.testing as npt
import pytest
from pytest import RaisesExc, RaisesGroup

import rlic
from rlic._histeq import MSG_EVEN, MSG_TOO_LOW, Strategy, as_pair

if sys.version_info >= (3, 13):
    from copy import replace as copy_replace
else:
    from dataclasses import replace as copy_replace


@pytest.mark.parametrize("key", ["tile-size", "tile-size-max"])
def test_missing_strategy_kind(key):
    with pytest.raises(TypeError, match=r"^strategy dict is missing a 'kind' key\.$"):
        Strategy.from_spec({key: 3})


def test_empty_spec():
    with RaisesGroup(
        RaisesExc(TypeError, match=r"^strategy dict is missing a 'kind' key\.$"),
        RaisesExc(
            TypeError,
            match=r"^Neither 'tile-size' nor 'tile-size-max' keys were found\. Either are allowed, but exactly one is expected\.$",
        ),
    ):
        Strategy.from_spec({})


def test_unknown_strategy_kind():
    with pytest.raises(
        ValueError,
        match=(
            r"^Unknown strategy kind 'not-a-kind'\. "
            r"Expected one of \['sliding-tile'\]$"
        ),
    ):
        Strategy.from_spec({"kind": "not-a-kind", "tile-size-max": 3})


def test_sliding_tile_missing_tile_size():
    with pytest.raises(
        TypeError,
        match=(
            r"Neither 'tile-size' nor 'tile-size-max' keys were found\. "
            r"Either are allowed, but exactly one is expected\."
        ),
    ):
        Strategy.from_spec({"kind": "sliding-tile"})


@pytest.mark.parametrize("key", ["tile-size", "tile-size-max"])
def test_sliding_tile_invalid_type_tile_size(key):
    with pytest.raises(
        TypeError,
        match=(
            rf"Incorrect type associated with key {key!r}. "
            r"Received 1\.5 with type <class 'float'>\. "
            r"Expected a single int, or a pair thereof\."
        ),
    ):
        Strategy.from_spec({"kind": "sliding-tile", key: 1.5})


@pytest.mark.parametrize(
    "tile_shape_max, containing_shape, expected_shape",
    [
        ((-1, -1), (5, 7), (5, 7)),
        ((-1, 3), (5, 7), (5, 3)),
        ((3, -1), (5, 7), (3, 7)),
        # even values are allowed in containing shapes, but padded in the output
        ((-1, -1), (6, 8), (7, 9)),
        ((-1, 3), (6, 8), (7, 3)),
        ((3, -1), (6, 8), (3, 9)),
    ],
)
def test_strategy_resolve_shape(tile_shape_max, containing_shape, expected_shape):
    s0 = Strategy(kind="sliding-window", tile_shape_max=tile_shape_max)
    s1 = s0.resolve_tile_shape(containing_shape)
    assert s1 is not s0
    assert s1.tile_shape_max == expected_shape
    assert copy_replace(s0, tile_shape_max=s1.tile_shape_max) == s1


def test_sliding_tile_both_tile_sizes_keys():
    with pytest.raises(
        TypeError,
        match=(
            r"Both 'tile-size' and 'tile-size-max' keys were provided\. "
            r"Only one of them can be specified at a time\."
        ),
    ):
        Strategy.from_spec(
            {"kind": "sliding-tile", "tile-size": 11, "tile-size-max": 13}
        )


@pytest.mark.parametrize(
    "key, prefix", [("tile-size", ""), ("tile-size-max", "Maximum ")]
)
@pytest.mark.parametrize(
    "size, axis, msg",
    [
        ((1, 3), "x", MSG_TOO_LOW),
        ((3, 1), "y", MSG_TOO_LOW),
        ((4, 3), "x", MSG_EVEN),
        ((3, 4), "y", MSG_EVEN),
    ],
)
def test_sliding_single_invalid_tile_size_value(key, size, prefix, axis, msg):
    if axis == "x":
        s = size[0]
    else:
        s = size[1]
    with pytest.raises(
        ValueError,
        match=re.escape(msg.format(prefix=prefix, axis=axis, size=s)),
    ):
        Strategy.from_spec({"kind": "sliding-tile", key: size})


@pytest.mark.parametrize(
    "key, prefix", [("tile-size", ""), ("tile-size-max", "Maximum ")]
)
@pytest.mark.parametrize(
    "size, msg",
    [
        (1, MSG_TOO_LOW),
        ((1, 1), MSG_TOO_LOW),
        (4, MSG_EVEN),
    ],
)
def test_sliding_two_invalid_tile_size_value(key, size, prefix, msg):
    ps = as_pair(size)
    with RaisesGroup(
        RaisesExc(
            ValueError,
            match=re.escape(msg.format(prefix=prefix, axis="x", size=ps[0])),
        ),
        RaisesExc(
            ValueError,
            match=re.escape(msg.format(prefix=prefix, axis="y", size=ps[1])),
        ),
    ):
        Strategy.from_spec({"kind": "sliding-tile", key: size})


@pytest.mark.parametrize("spec", [-2, -1, (-1, -1), (5, -1), (-1, 5)])
def test_stragegy_from_spec_negative_max_size(spec):
    strat = Strategy.from_spec({"kind": "sliding-tile", "tile-size-max": spec})
    assert strat.tile_shape_max == as_pair(spec)


@pytest.mark.parametrize(
    "spec, expected",
    [
        pytest.param(
            {"kind": "sliding-tile", "tile-size": 13},
            Strategy(kind="sliding-tile", tile_shape=(13, 13)),
            id="int-tile-size",
        ),
        pytest.param(
            {"kind": "sliding-tile", "tile-size": (13, 15)},
            Strategy(kind="sliding-tile", tile_shape=(13, 15)),
            id="tuple-tile-size",
        ),
        pytest.param(
            {"kind": "sliding-tile", "tile-size-max": 13},
            Strategy(kind="sliding-tile", tile_shape_max=(13, 13)),
            id="int-tile-size-max",
        ),
        pytest.param(
            {"kind": "sliding-tile", "tile-size-max": (13, 15)},
            Strategy(kind="sliding-tile", tile_shape_max=(13, 15)),
            id="tuple-tile-size-max",
        ),
    ],
)
def test_sliding_tile_from_spec(spec, expected):
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
