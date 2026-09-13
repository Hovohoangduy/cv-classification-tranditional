#!/usr/bin/env python3
"""Inference entry point for trained CIFAR-10 Pixel/HOG classifiers."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_cifar10
from src.features import HOG
from src.preprocessing import to_float


def load_image(path: str | Path) -> np.ndarray:
    """Load an arbitrary image and convert it to CIFAR-10 RGB shape."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def extract_features(
    images: np.ndarray,
    feature_name: str,
    feature_parameters: dict[str, Any] | None = None,
) -> np.ndarray:
    parameters = feature_parameters or {}
    if feature_name == "pixel":
        return to_float(images).reshape(len(images), -1)
    if feature_name == "hog":
        extractor = HOG(
            cell_size=int(parameters.get("cell_size", 4)),
            block_size=int(parameters.get("block_size", 2)),
            num_bins=int(parameters.get("num_bins", 9)),
        )
        return extractor.transform(images)
    raise ValueError(
        f"Feature '{feature_name}' chưa hỗ trợ inference ảnh rời. "
        "Hiện inference hỗ trợ pixel và HOG."
    )


def predict_images(
    bundle: dict[str, Any],
    images: np.ndarray,
    feature_override: str | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    feature_name = feature_override or bundle.get("feature_name")
    if feature_name is None:
        raise ValueError("Model cũ không lưu feature_name; hãy truyền --feature pixel hoặc --feature hog")
    features = extract_features(images, str(feature_name), bundle.get("feature_parameters"))
    scaled = bundle["scaler"].transform(features)
    model = bundle["model"]
    predictions = model.predict(scaled)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(scaled)
    else:
        scores = np.zeros((len(images), len(bundle["class_names"])), dtype=np.float32)
    class_names = np.asarray(bundle["class_names"])
    results: list[dict[str, Any]] = []
    top_k = max(1, min(int(top_k), len(class_names)))
    for index, predicted_label in enumerate(predictions):
        order = np.argsort(scores[index])[::-1][:top_k]
        results.append(
            {
                "predicted_label": int(predicted_label),
                "predicted_class": str(class_names[predicted_label]),
                "top_k": [
                    {"label": int(label), "class": str(class_names[label]), "score": float(scores[index, label])}
                    for label in order
                ],
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/models/hog_svm.pkl", help="Model bundle tạo bởi scripts/train.py")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", nargs="+", help="Một hoặc nhiều đường dẫn ảnh")
    inputs.add_argument("--cifar-index", nargs="+", type=int, help="Một hoặc nhiều index trong CIFAR-10 test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature", choices=("pixel", "hog"), help="Override cho model bundle cũ")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", help="Tùy chọn: lưu kết quả JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.model).open("rb") as stream:
        bundle = pickle.load(stream)

    sources: list[str]
    true_labels: list[int | None]
    if args.image:
        images = np.stack([load_image(path) for path in args.image])
        sources = [str(Path(path)) for path in args.image]
        true_labels = [None] * len(images)
    else:
        dataset = load_cifar10(args.data_dir, download=False)
        indices = np.asarray(args.cifar_index, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(dataset.test_images)):
            raise ValueError(f"CIFAR index phải nằm trong [0, {len(dataset.test_images) - 1}]")
        images = dataset.test_images[indices]
        sources = [f"cifar10:test[{index}]" for index in indices]
        true_labels = [int(label) for label in dataset.test_labels[indices]]

    results = predict_images(bundle, images, args.feature, args.top_k)
    class_names = np.asarray(bundle["class_names"])
    for result, source, true_label in zip(results, sources, true_labels):
        result["source"] = source
        result["true_label"] = true_label
        result["true_class"] = None if true_label is None else str(class_names[true_label])
        status = ""
        if true_label is not None:
            status = "ĐÚNG" if true_label == result["predicted_label"] else "SAI"
            status = f" | thật={result['true_class']} | {status}"
        top_text = ", ".join(f"{item['class']}:{item['score']:.3f}" for item in result["top_k"])
        print(f"{source} -> dự đoán={result['predicted_class']}{status} | top-{len(result['top_k'])}: {top_text}")

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Đã lưu kết quả vào {destination}")


if __name__ == "__main__":
    main()
