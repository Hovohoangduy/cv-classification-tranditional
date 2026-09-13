import numpy as np

from src.features.bovw import BagOfVisualWords, KMeans
from src.features.sift import SIFT, gaussian_blur


def test_gaussian_blur_preserves_constant_image():
    image = np.ones((20, 20), dtype=np.float32)
    assert np.allclose(gaussian_blur(image, 1.2), 1.0, atol=1e-5)


def test_sift_descriptor_has_128_dimensions():
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 1.0
    _, descriptors = SIFT(contrast_threshold=0.001).detect_and_describe(image)
    assert descriptors.ndim == 2
    assert descriptors.shape[1] == 128
    assert np.isfinite(descriptors).all()


def test_kmeans_and_bovw_shapes():
    rng = np.random.default_rng(0)
    descriptors = [rng.normal(size=(8, 4)).astype(np.float32) for _ in range(5)]
    model = BagOfVisualWords(vocabulary_size=3, max_iterations=10, seed=0)
    encoded = model.fit(descriptors).transform(descriptors)
    assert encoded.shape == (5, 3)
    assert np.allclose(np.linalg.norm(encoded, axis=1), 1.0)

