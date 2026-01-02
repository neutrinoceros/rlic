import numpy as np
import pytest

import rlic


@pytest.mark.parametrize(
    "bins, min_rms_reduction",
    [
        (12, 2.0),
        (64, 10.0),
        (256, 50.0),
    ],
)
def test_historgram_equalization(bins, min_rms_reduction):
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
    )

    def normalized_cdf(a):
        hist, bin_edges = np.histogram(a.ravel(), bins=bins)
        cdf = hist.cumsum()
        return cdf / float(cdf.max())

    def rms(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    image_eq = rlic.equalize_histogram(image, bins=bins)
    cdf_in = normalized_cdf(image)
    cdf_eq = normalized_cdf(image_eq)

    id_func = np.linspace(0, 1, bins)
    rms_in = rms(cdf_in, id_func)
    rms_eq = rms(cdf_eq, id_func)
    assert rms_eq < rms_in
    assert (rms_in / rms_eq) > min_rms_reduction
