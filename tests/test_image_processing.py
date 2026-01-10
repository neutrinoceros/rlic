import re
import sys
from itertools import product

import numpy as np
import numpy.testing as npt
import pytest
from pytest import RaisesExc, RaisesGroup

import rlic
from rlic._histeq import (
    MSG_EVEN,
    MSG_INTO_NEG,
    MSG_TOO_LOW,
    MSG_TOO_LOW_NEG,
    SlidingTile,
    TileInterpolation,
    as_pair,
)

if sys.version_info >= (3, 13):
    from copy import replace as copy_replace
else:
    from dataclasses import replace as copy_replace


def test_sliding_tile_from_spec_missing_tile_size():
    with pytest.raises(
        TypeError,
        match=(
            r"Sliding tile specification is missing a 'tile-size-max' key\. "
            r"Expected a single int, or a pair thereof\."
        ),
    ):
        SlidingTile.from_spec({"kind": "sliding-tile"})


def test_sliding_tile_from_spec_invalid_type_tile_size():
    with pytest.raises(
        TypeError,
        match=(
            r"Incorrect type associated with key 'tile-size-max'\. "
            r"Received 1\.5 with type <class 'float'>\. "
            r"Expected a single int, or a pair thereof\."
        ),
    ):
        SlidingTile.from_spec({"kind": "sliding-tile", "tile-size-max": 1.5})


@pytest.mark.parametrize(
    "tile_shape_max, image_shape, expected_shape",
    [
        ((-1, -1), (5, 7), (5, 7)),
        ((-1, 3), (5, 7), (5, 3)),
        ((3, -1), (5, 7), (3, 7)),
        # even values are allowed in containing shapes, but padded in the output
        ((-1, -1), (6, 8), (7, 9)),
        ((-1, 3), (6, 8), (7, 3)),
        ((3, -1), (6, 8), (3, 9)),
        # negative values are special cased:
        # -1 means "match the containing shape"
        # -2 means "double the containing shape"
        ((-2, -2), (5, 7), (11, 15)),
    ],
)
def test_sliding_tile_resolve(tile_shape_max, image_shape, expected_shape):
    s0 = SlidingTile(tile_shape_max=tile_shape_max)
    s1 = s0.resolve(image_shape=image_shape)
    assert s1 is not s0
    assert s1.tile_shape_max == expected_shape
    assert copy_replace(s0, tile_shape_max=s1.tile_shape_max) == s1


@pytest.mark.parametrize(
    "cls, kind",
    [(SlidingTile, "sliding-tile"), (TileInterpolation, "tile-interpolation")],
)
@pytest.mark.parametrize(
    "size, axis, msg",
    [
        ((-3, 3), "x", MSG_TOO_LOW_NEG),
        ((3, -3), "y", MSG_TOO_LOW_NEG),
        ((1, 3), "x", MSG_TOO_LOW),
        ((3, 1), "y", MSG_TOO_LOW),
        ((4, 3), "x", MSG_EVEN),
        ((3, 4), "y", MSG_EVEN),
    ],
)
def test_from_spec_single_invalid_tile_size_value(cls, kind, size, axis, msg):
    if axis == "x":
        s = size[0]
    else:
        s = size[1]
    with pytest.raises(
        ValueError,
        match=re.escape(msg.format(axis=axis, size=s)),
    ):
        cls.from_spec({"kind": kind, "tile-size-max": size})


@pytest.mark.parametrize(
    "cls, kind",
    [(SlidingTile, "sliding-tile"), (TileInterpolation, "tile-interpolation")],
)
@pytest.mark.parametrize(
    "size, msg",
    [
        (1, MSG_TOO_LOW),
        ((1, 1), MSG_TOO_LOW),
        (4, MSG_EVEN),
    ],
)
def test_from_spec_invalid_tile_size_max_value(cls, kind, size, msg):
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
        cls.from_spec({"kind": kind, "tile-size-max": size})


@pytest.mark.parametrize("spec", [-2, -1, 0])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_tile_interpolation_from_spec_single_invalid_tile_into_value(spec, axis):
    if axis == "x":
        into = (spec, 1)
    else:
        into = (1, spec)
    with pytest.raises(
        ValueError,
        match=re.escape(MSG_INTO_NEG.format(axis=axis, into=spec)),
    ):
        TileInterpolation.from_spec({"kind": "tile-interpolation", "tile-into": into})


@pytest.mark.parametrize("spec", [-2, -1, (-1, -1), (5, -1), (-1, 5)])
def test_sliding_tile_from_spec_negative_tile_size_max(spec):
    strat = SlidingTile.from_spec({"kind": "sliding-tile", "tile-size-max": spec})
    assert strat.tile_shape_max == as_pair(spec)


@pytest.mark.parametrize(
    "spec, expected",
    [
        pytest.param(
            {"kind": "sliding-tile", "tile-size-max": 13},
            SlidingTile(tile_shape_max=(13, 13)),
            id="int-tile-size-max",
        ),
        pytest.param(
            {"kind": "sliding-tile", "tile-size-max": (13, 15)},
            SlidingTile(tile_shape_max=(13, 15)),
            id="tuple-tile-size-max",
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
            r"Neither 'tile-into' nor 'tile-size-max' keys were found\. "
            r"Either are allowed, but exactly one is expected\."
        ),
    ):
        TileInterpolation.from_spec({"kind": "tile-interpolation"})


def test_tile_interpolation_from_spec_both_tile_spec_keys():
    with pytest.raises(
        TypeError,
        match=(
            r"Both 'tile-into' and 'tile-size-max' keys were provided\. "
            r"Only one of them can be specified at a time\."
        ),
    ):
        TileInterpolation.from_spec(
            {"kind": "tile-interpolation", "tile-into": 11, "tile-size-max": 13}
        )


@pytest.mark.parametrize("key", ["tile-into", "tile-size-max"])
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
            {"kind": "tile-interpolation", "tile-size-max": 13},
            TileInterpolation(tile_shape_max=(13, 13)),
            id="int-tile-size-max",
        ),
        pytest.param(
            {"kind": "tile-interpolation", "tile-size-max": (13, 15)},
            TileInterpolation(tile_shape_max=(13, 15)),
            id="tuple-tile-size-max",
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


def _get_tile(image, center_pixel: tuple[int, int], max_size: int):
    wing_size = max_size // 2
    isize, jsize = image.shape
    i0, j0 = center_pixel
    iL = max(0, i0 - wing_size)
    iR = min(i0 + wing_size, isize - 1)
    jL = max(0, j0 - wing_size)
    jR = min(j0 + wing_size, jsize - 1)
    return image[iL : iR + 1, jL : jR + 1]


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


def _ahe_numpy(image, nbins: int, tile_size_max: int):
    out = np.zeros_like(image)
    for i, j in product(*(range(size) for size in image.shape)):
        tile = _get_tile(image, center_pixel=(i, j), max_size=tile_size_max)
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


@pytest.mark.parametrize("key", ["tile-into", "tile-size-max"])
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


@pytest.mark.parametrize("nbins", [12, 64, 256, "auto"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_historgram_equalization_sliding_tile_full_image(nbins, dtype, subtests):
    IMAGE_SHAPE = (256, 128)
    prng = np.random.default_rng(0)
    image = np.clip(
        prng.normal(loc=5.0, scale=1.0, size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE),
        a_min=0.0,
        a_max=None,
        dtype=dtype,
    )

    resolved_nbins = 256 if nbins == "auto" else nbins

    res_default = rlic.equalize_histogram(
        image,
        nbins=resolved_nbins,
        adaptive_strategy=None,
    )

    res_st = rlic.equalize_histogram(
        image,
        nbins=nbins,
        adaptive_strategy={
            "kind": "sliding-tile",
            # use a sliding tile that always contains the entire
            # image to help comparing with the non-adaptive case
            "tile-size-max": -2,
        },
    )
    npt.assert_array_equal(res_st, res_default)


def test_historgram_equalization_sliding_tile_full_ahe():
    IMAGE_SHAPE = (4, 6)
    TILE_SIZE_MAX = 3
    NBINS = 3
    prng = np.random.default_rng(0)
    image = np.clip(
        prng.normal(loc=5.0, scale=1.0, size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE),
        a_min=0.0,
        a_max=None,
        dtype="float64",
    )

    res_ahe = _ahe_numpy(image, nbins=NBINS, tile_size_max=TILE_SIZE_MAX)

    res_st = rlic.equalize_histogram(
        image,
        nbins=NBINS,
        adaptive_strategy={
            "kind": "sliding-tile",
            "tile-size-max": TILE_SIZE_MAX,
        },
    )
    npt.assert_allclose(res_st, res_ahe, rtol=5e-16)
