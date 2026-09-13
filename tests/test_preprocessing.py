import numpy as np

from src.preprocessing import StandardScaler, rgb_to_gray


def test_scaler_handles_constant_columns():
    x = np.array([[1, 2], [1, 4]], dtype=np.float32)
    transformed = StandardScaler().fit_transform(x)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed[:, 0], 0)


def test_rgb_to_gray_shape():
    images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    assert rgb_to_gray(images).shape == (3, 32, 32)

