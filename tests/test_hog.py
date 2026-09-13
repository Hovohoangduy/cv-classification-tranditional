import numpy as np

from src.features.hog import HOG


def test_hog_has_expected_dimension_and_no_nan():
    image = np.zeros((32, 32), dtype=np.float32)
    descriptor = HOG(cell_size=4, block_size=2, num_bins=9).transform_one(image)
    assert descriptor.shape == (7 * 7 * 2 * 2 * 9,)
    assert np.isfinite(descriptor).all()
    assert np.allclose(descriptor, 0)


def test_hog_detects_horizontal_intensity_gradient():
    image = np.tile(np.linspace(0, 1, 32), (32, 1)).astype(np.float32)
    descriptor = HOG(cell_size=8, block_size=1, num_bins=9).transform_one(image)
    histogram = descriptor.reshape(-1, 9).sum(axis=0)
    assert np.argmax(histogram) == 0

