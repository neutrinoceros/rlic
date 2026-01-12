import re
from itertools import product

import numpy as np
import numpy.testing as npt
import pytest
from pytest import RaisesExc, RaisesGroup

import rlic
from rlic._histeq import (
    INTO_NEG,
    TS_EVEN,
    TS_INVALID_EXPECTED_EVEN,
    TS_INVALID_EXPECTED_ODD,
    TS_ODD,
    SlidingTile,
    TileInterpolation,
    as_pair,
    minimal_divisor_size,
)


def test_sliding_tile_from_spec_missing_tile_size():
    with pytest.raises(
        TypeError,
        match=(
            r"Sliding tile specification is missing a 'tile-size' key\. "
            r"Expected a single int, or a pair thereof\."
        ),
    ):
        SlidingTile.from_spec({"kind": "sliding-tile"})


def test_sliding_tile_from_spec_invalid_type_tile_size():
    with pytest.raises(
        TypeError,
        match=(
            r"Incorrect type associated with key 'tile-size'\. "
            r"Received 1\.5 with type <class 'float'>\. "
            r"Expected a single int, or a pair thereof\."
        ),
    ):
        SlidingTile.from_spec({"kind": "sliding-tile", "tile-size": 1.5})


@pytest.mark.parametrize(
    "tile_shape, image_shape, expected_shape",
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
def test_sliding_tile_resolve(tile_shape, image_shape, expected_shape):
    st = SlidingTile(tile_shape=tile_shape)
    assert st.resolve_tile_shape(image_shape) == expected_shape


@pytest.mark.parametrize(
    "size, axis, msg",
    [
        ((-3, 3), "x", TS_INVALID_EXPECTED_ODD),
        ((3, -3), "y", TS_INVALID_EXPECTED_ODD),
        ((1, 3), "x", TS_INVALID_EXPECTED_ODD),
        ((3, 1), "y", TS_INVALID_EXPECTED_ODD),
        ((4, 3), "x", TS_EVEN),
        ((3, 4), "y", TS_EVEN),
    ],
)
def test_sliding_tile_from_spec_single_invalid_tile_size_value(size, axis, msg):
    if axis == "x":
        s = size[0]
    else:
        s = size[1]
    with pytest.raises(
        ValueError,
        match=re.escape(msg.format(axis=axis, size=s)),
    ):
        SlidingTile.from_spec({"kind": "sliding-tile", "tile-size": size})


@pytest.mark.parametrize(
    "size, axis, msg",
    [
        ((-4, 4), "x", TS_INVALID_EXPECTED_EVEN),
        ((4, -4), "y", TS_INVALID_EXPECTED_EVEN),
        ((3, 4), "x", TS_ODD),
        ((4, 3), "y", TS_ODD),
    ],
)
def test_tile_interpolation_from_spec_single_invalid_tile_size_value(size, axis, msg):
    if axis == "x":
        s = size[0]
    else:
        s = size[1]
    with pytest.raises(
        ValueError,
        match=re.escape(msg.format(axis=axis, size=s)),
    ):
        TileInterpolation.from_spec({"kind": "tile-interpolation", "tile-size": size})


@pytest.mark.parametrize(
    "size, msg",
    [
        (1, TS_INVALID_EXPECTED_ODD),
        ((1, 1), TS_INVALID_EXPECTED_ODD),
        (4, TS_EVEN),
    ],
)
def test_sliding_tile_from_spec_invalid_tile_size_value(size, msg):
    ps = as_pair(size)
    with RaisesGroup(
        RaisesExc(
            ValueError,
            match=re.escape(msg.format(axis="x", size=ps[0])),
        ),
        RaisesExc(
            ValueError,
            match=re.escape(msg.format(axis="y", size=ps[1])),
        ),
    ):
        SlidingTile.from_spec({"kind": "sliding-tile", "tile-size": size})


@pytest.mark.parametrize(
    "size, msg",
    [
        (0, TS_INVALID_EXPECTED_EVEN),
        ((0, 0), TS_INVALID_EXPECTED_EVEN),
        (3, TS_ODD),
    ],
)
def test_tile_interpolation_from_spec_invalid_tile_size_value(size, msg):
    ps = as_pair(size)
    with RaisesGroup(
        RaisesExc(
            ValueError,
            match=re.escape(msg.format(axis="x", size=ps[0])),
        ),
        RaisesExc(
            ValueError,
            match=re.escape(msg.format(axis="y", size=ps[1])),
        ),
    ):
        TileInterpolation.from_spec({"kind": "tile-interpolation", "tile-size": size})


@pytest.mark.parametrize("spec", [-2, -1, 0])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_tile_interpolation_from_spec_single_invalid_tile_into_value(spec, axis):
    if axis == "x":
        into = (spec, 1)
    else:
        into = (1, spec)
    with pytest.raises(
        ValueError,
        match=re.escape(INTO_NEG.format(axis=axis, into=spec)),
    ):
        TileInterpolation.from_spec({"kind": "tile-interpolation", "tile-into": into})


@pytest.mark.parametrize("spec", [-1, (-1, -1), (5, -1), (-1, 5)])
def test_sliding_tile_from_spec_negative_tile_size(spec):
    strat = SlidingTile.from_spec({"kind": "sliding-tile", "tile-size": spec})
    assert strat.tile_shape == as_pair(spec)


@pytest.mark.parametrize(
    "spec, expected",
    [
        pytest.param(
            {"kind": "sliding-tile", "tile-size": 13},
            SlidingTile(tile_shape=(13, 13)),
            id="int-tile-size",
        ),
        pytest.param(
            {"kind": "sliding-tile", "tile-size": (13, 15)},
            SlidingTile(tile_shape=(13, 15)),
            id="tuple-tile-size",
        ),
    ],
)
def test_sliding_tile_from_spec(spec, expected):
    strategy = SlidingTile.from_spec(spec)
    assert strategy == expected


def test_tile_interpolation_from_spec_no_tile_spec():
    with pytest.raises(
        TypeError,
        match=(
            r"Neither 'tile-into' nor 'tile-size' keys were found\. "
            r"Either are allowed, but exactly one is expected\."
        ),
    ):
        TileInterpolation.from_spec({"kind": "tile-interpolation"})


def test_tile_interpolation_from_spec_both_tile_spec_keys():
    with pytest.raises(
        TypeError,
        match=(
            r"Both 'tile-into' and 'tile-size' keys were provided\. "
            r"Only one of them can be specified at a time\."
        ),
    ):
        TileInterpolation.from_spec(
            {"kind": "tile-interpolation", "tile-into": 11, "tile-size": 13}
        )


@pytest.mark.parametrize("key", ["tile-into", "tile-size"])
def test_tile_interpolation_from_spec_invalid_type(key):
    with pytest.raises(
        TypeError,
        match=(
            f"Incorrect type associated with key {key!r}. "
            f"Received None with type <class 'NoneType'>. "
            "Expected a single int, or a pair thereof."
        ),
    ):
        TileInterpolation.from_spec({"kind": "tile-interpolation", key: None})


@pytest.mark.parametrize(
    "spec, expected",
    [
        pytest.param(
            {"kind": "tile-interpolation", "tile-size": 14},
            TileInterpolation(tile_shape=(14, 14)),
            id="int-tile-size",
        ),
        pytest.param(
            {"kind": "tile-interpolation", "tile-size": (12, 14)},
            TileInterpolation(tile_shape=(12, 14)),
            id="tuple-tile-size",
        ),
        pytest.param(
            {"kind": "tile-interpolation", "tile-into": 4},
            TileInterpolation(tile_into=(4, 4)),
            id="int-tile-into",
        ),
        pytest.param(
            {"kind": "tile-interpolation", "tile-into": (2, 3)},
            TileInterpolation(tile_into=(2, 3)),
            id="tuple-tile-into",
        ),
    ],
)
def test_tile_interpolation_from_spec(spec, expected):
    strategy = TileInterpolation.from_spec(spec)
    assert strategy == expected


@pytest.mark.parametrize(
    "strat, image_shape, expected_shape",
    [
        (TileInterpolation(tile_into=(2, 2)), (256, 256), (128, 128)),
        (TileInterpolation(tile_into=(2, 3)), (256, 256), (128, 86)),
        (TileInterpolation(tile_into=(5, 1)), (256, 256), (52, 256)),
        (TileInterpolation(tile_shape=(64, 64)), (256, 256), (64, 64)),
    ],
)
def test_tile_interpolation_resolve(strat, image_shape, expected_shape):
    assert strat.resolve_tile_shape(image_shape) == expected_shape


@pytest.mark.parametrize(
    "nbins, shape, expected",
    [
        (32, (8, 8), 32),
        (64, (8, 8), 64),
        (256, (8, 8), 256),
        ("auto", (8, 8), 64),
        ("auto", (4, 4), 16),
        ("auto", (1024, 2048), 256),
    ],
)
def test_auto_nbins(nbins, shape, expected):
    res = rlic._lib._resolve_nbins(nbins, shape)
    assert res == expected


def _get_tile(image, center_pixel: tuple[int, int], max_size: int):
    wing_size = max_size // 2
    isize, jsize = image.shape
    i0, j0 = center_pixel

    pimage = np.pad(image, pad_width=wing_size, mode="reflect")
    iL = i0
    iR = i0 + 2 * wing_size
    jL = j0
    jR = j0 + 2 * wing_size
    return pimage[iL : iR + 1, jL : jR + 1]


def _get_normalized_cdf(tile, nbins: int):
    hist, bin_edges = np.histogram(tile.ravel(), bins=nbins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # cumulative distribution function
    cdf = hist.cumsum(dtype=tile.dtype)
    return (bin_centers, cdf / cdf[-1])


def _equalize_histogram_numpy(image, nbins: int):
    # ref implementation, adapted from scikit-image (exposure.equalize_hist)
    bin_centers, normalized_cdf = _get_normalized_cdf(image, nbins)

    # As of version 2.4, np.interp always promotes to float64, so we
    # have to cast back to single precision when float32 output is desired
    return (
        np.interp(image.flat, bin_centers, normalized_cdf)
        .reshape(image.shape)
        .astype(image.dtype, copy=False)
    )


def _ahe_numpy(image, nbins: int, tile_size: int):
    out = np.zeros_like(image)
    for i, j in product(*(range(size) for size in image.shape)):
        tile = _get_tile(image, center_pixel=(i, j), max_size=tile_size)
        bin_centers, normalized_cdf = _get_normalized_cdf(tile, nbins)
        out[i, j] = np.interp(image[i, j], bin_centers, normalized_cdf)
    return out


@pytest.mark.parametrize(
    "nbins, min_rms_reduction",
    [
        (12, 2.0),
        (64, 10.0),
        (256, 50.0),
    ],
)
@pytest.mark.parametrize("dtype, rtol", [("float32", 2e-6), ("float64", 2.5e-15)])
def test_historgram_equalization(nbins, min_rms_reduction, dtype, rtol, subtests):
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

    with subtests.test("compare to ref impl"):
        npt.assert_allclose(image_eq, image_ref, rtol=rtol)


@pytest.mark.parametrize("key", ["tile-into", "tile-size"])
def test_equalize_histogram_missing_strategy_kind(key):
    with pytest.raises(
        TypeError, match=r"^adaptive_strategy is missing a 'kind' key\.$"
    ):
        rlic.equalize_histogram(
            np.eye(5, dtype="float64"),
            adaptive_strategy={},
        )


def test_equalize_histogram_unknown_strategy_kind():
    with pytest.raises(
        ValueError,
        match=(
            r"^Unknown strategy kind 'not-a-kind'\. "
            r"Expected one of \['sliding-tile', 'tile-interpolation'\]$"
        ),
    ):
        rlic.equalize_histogram(
            np.eye(5, dtype="float64"),
            adaptive_strategy={"kind": "not-a-kind"},
        )


def test_equalize_histogram_invalid_strategy_kind_type():
    with pytest.raises(
        TypeError,
        match=(
            r"^Invalid strategy kind None with type <class 'NoneType'>\. "
            r"Expected one of \['sliding-tile', 'tile-interpolation'\]$"
        ),
    ):
        rlic.equalize_histogram(
            np.eye(5, dtype="float64"),
            adaptive_strategy={"kind": None},
        )


def test_historgram_equalization_unsupported_dtype():
    with pytest.raises(
        TypeError,
        match=(
            r"^Found unsupported data type: int64\. "
            r"Expected of of \[dtype\('float32'\), dtype\('float64'\)\]\.$"
        ),
    ):
        rlic.equalize_histogram(np.eye(3, dtype="int64"), nbins=3)


@pytest.mark.parametrize("dtype, rtol", [("float32", 6e-7), ("float64", 5e-16)])
def test_historgram_equalization_sliding_tile_full_ahe(dtype, rtol):
    IMAGE_SHAPE = (4, 6)
    TILE_SIZE = 3
    NBINS = 3
    prng = np.random.default_rng(0)
    image = np.clip(
        prng.normal(loc=5.0, scale=1.0, size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE),
        a_min=0.0,
        a_max=None,
        dtype=dtype,
    )

    res_ahe = _ahe_numpy(image, nbins=NBINS, tile_size=TILE_SIZE)

    res_st = rlic.equalize_histogram(
        image,
        nbins=NBINS,
        adaptive_strategy={
            "kind": "sliding-tile",
            "tile-size": TILE_SIZE,
        },
    )
    npt.assert_allclose(res_st, res_ahe, rtol=rtol)


@pytest.mark.parametrize("dtype, rtol", [("float32", 1.25), ("float64", 1.25)])
def test_historgram_equalization_tile_interpolation_full_ahe(dtype, rtol):
    IMAGE_SHAPE = (64, 64)
    TILE_SIZE = 32
    NBINS = 3
    prng = np.random.default_rng(0)
    image = np.clip(
        prng.normal(loc=5.0, scale=1.0, size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE),
        a_min=0.0,
        a_max=None,
        dtype=dtype,
    )

    res_ahe = _ahe_numpy(image, nbins=NBINS, tile_size=TILE_SIZE)

    res_st = rlic.equalize_histogram(
        image,
        nbins=NBINS,
        adaptive_strategy={
            "kind": "tile-interpolation",
            "tile-size": TILE_SIZE,
        },
    )
    npt.assert_allclose(res_st, res_ahe, rtol=rtol)


@pytest.mark.parametrize(
    "size, into, expected",
    [
        (3, 1, 3),
        (3, 2, 2),
        (3, 3, 1),
        (4, 1, 4),
        (4, 2, 2),
        (4, 3, 2),
        (4, 4, 1),
        (5, 1, 5),
        (5, 2, 3),
        (5, 3, 2),
        (5, 4, 2),
        (5, 5, 1),
    ],
)
def test_minimal_size_divisor(size, into, expected):
    assert minimal_divisor_size(size, into) == expected
