import numpy as np

from PIL import Image

from inference import extract_features, load_image, predict_images
from src.models import LinearSVM
from src.preprocessing import StandardScaler


def test_extract_hog_features_has_expected_shape():
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    features = extract_features(images, "hog", {"cell_size": 4, "block_size": 2, "num_bins": 9})
    assert features.shape == (2, 1764)


def test_load_image_converts_to_cifar_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (19, 41), color=127).save(path)
    image = load_image(path)
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8


def test_predict_images_returns_top_k():
    dark = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    bright = np.full((10, 32, 32, 3), 255, dtype=np.uint8)
    images = np.concatenate([dark, bright])
    labels = np.repeat([0, 1], 10)
    raw = extract_features(images, "pixel")
    scaler = StandardScaler().fit(raw)
    model = LinearSVM(learning_rate=0.1, epochs=10, batch_size=4, seed=1).fit(scaler.transform(raw), labels)
    bundle = {
        "model": model,
        "scaler": scaler,
        "class_names": np.asarray(["dark", "bright"]),
        "feature_name": "pixel",
        "feature_parameters": {},
    }
    results = predict_images(bundle, images[[0, -1]], top_k=2)
    assert [result["predicted_class"] for result in results] == ["dark", "bright"]
    assert all(len(result["top_k"]) == 2 for result in results)
