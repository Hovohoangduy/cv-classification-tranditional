"""Kiểm thử gọn cho toàn bộ thuật toán cốt lõi và inference."""

import pickle

import numpy as np
from PIL import Image

from src.data import StandardScaler, rgb_to_gray, stratified_split
from src.features import BagOfVisualWords, HOG, SIFT, gaussian_blur
from src.models import DecisionTreeClassifier, LinearSVM, RandomForestClassifier, load_bundle
from src.pipeline import classification_metrics
from inference import actual_labels_for_images, extract_features, load_image, plot_predictions, predict_images


def test_data_and_preprocessing():
    labels = np.repeat(np.arange(3), 20)
    train, validation = stratified_split(labels, 0.2, seed=7)
    assert not set(train) & set(validation)
    assert np.bincount(labels[validation]).tolist() == [4, 4, 4]
    values = StandardScaler().fit_transform(np.array([[1, 2], [1, 4]], dtype=np.float32))
    assert np.isfinite(values).all() and np.allclose(values[:, 0], 0)
    assert rgb_to_gray(np.zeros((3, 32, 32, 3), dtype=np.uint8)).shape == (3, 32, 32)


def test_hog_sift_and_bovw():
    image = np.zeros((32, 32), dtype=np.float32)
    hog = HOG().transform_one(image)
    assert hog.shape == (1764,) and np.isfinite(hog).all()
    assert np.allclose(gaussian_blur(np.ones((20, 20), dtype=np.float32), 1.2), 1.0, atol=1e-5)
    image[8:24, 8:24] = 1
    descriptors = SIFT(contrast_threshold=0.001).detect_and_describe(image)[1]
    assert descriptors.shape[1] == 128
    rng = np.random.default_rng(0)
    sets = [rng.normal(size=(8, 4)).astype(np.float32) for _ in range(5)]
    encoded = BagOfVisualWords(vocabulary_size=3, max_iterations=10, seed=0).fit_transform(sets)
    assert encoded.shape == (5, 3) and np.allclose(np.linalg.norm(encoded, axis=1), 1)


def test_models_and_metrics():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal((-2, -2), .2, (30, 2)), rng.normal((2, -2), .2, (30, 2)), rng.normal((0, 2), .2, (30, 2))]).astype(np.float32)
    y = np.repeat(np.arange(3), 30)
    svm = LinearSVM(learning_rate=.05, epochs=30, batch_size=15, seed=1).fit(x, y)
    assert np.mean(svm.predict(x) == y) > .98
    line = np.arange(20, dtype=np.float32)[:, None]
    binary = (line[:, 0] >= 10).astype(np.int64)
    tree = DecisionTreeClassifier(max_depth=2, max_features=1, seed=1).fit(line, binary)
    assert np.mean(tree.predict(line) == binary) == 1
    forest = RandomForestClassifier(num_trees=5, max_depth=5, seed=2).fit(x, y)
    assert set(np.unique(forest.predict(x))) <= {0, 1, 2}
    assert classification_metrics(y, y, 3)["macro_f1"] == 1


def test_inference_and_bundle(tmp_path):
    image_path = tmp_path / "gray.png"
    Image.new("L", (19, 41), color=127).save(image_path)
    assert load_image(image_path).shape == (32, 32, 3)
    dark = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    bright = np.full((10, 32, 32, 3), 255, dtype=np.uint8)
    images, labels = np.concatenate([dark, bright]), np.repeat([0, 1], 10)
    raw = extract_features(images, "pixel")
    scaler = StandardScaler().fit(raw)
    model = LinearSVM(learning_rate=.1, epochs=10, batch_size=4, seed=1).fit(scaler.transform(raw), labels)
    bundle = {"model": model, "scaler": scaler, "class_names": np.asarray(["dark", "bright"]), "feature_name": "pixel", "feature_parameters": {}}
    results = predict_images(bundle, images[[0, -1]], top_k=2)
    assert [item["predicted_class"] for item in results] == ["dark", "bright"]
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(bundle))
    assert load_bundle(path)["feature_name"] == "pixel"

    demo_path = tmp_path / "cat_00001.png"
    Image.new("RGB", (32, 32)).save(demo_path)
    manifest = tmp_path / "labels.json"
    manifest.write_text('{"images":{"cat_00001.png":{"label":1}}}', encoding="utf-8")
    assert actual_labels_for_images([str(demo_path)], ["dog", "cat"], manifest) == [1]
    results[0].update({"actual_label": 0, "actual_class": "dark"})
    plot_path = tmp_path / "prediction.png"
    assert plot_predictions(images[[0]], results[:1], plot_path) == plot_path
    assert plot_path.exists()
