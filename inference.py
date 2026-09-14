#!/usr/bin/env python3
"""Inference entry point for trained CIFAR-10 Pixel/HOG classifiers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_cifar10, to_float
from src.features import HOG
from src.models import load_bundle


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


def _parse_label(value: str, class_names: list[str]) -> int:
    try:
        label = int(value)
    except ValueError:
        if value not in class_names:
            choices = ", ".join(class_names)
            raise ValueError(f"Nhãn '{value}' không hợp lệ. Các lớp: {choices}") from None
        label = class_names.index(value)
    if not 0 <= label < len(class_names):
        raise ValueError(f"Nhãn số phải nằm trong [0, {len(class_names) - 1}]")
    return label


def actual_labels_for_images(
    image_paths: list[str],
    class_names: list[str],
    manifest_path: str | Path | None = None,
    explicit_labels: list[str] | None = None,
) -> list[int | None]:
    """Lấy nhãn thật do người dùng truyền hoặc từ manifest ảnh demo."""
    if explicit_labels is not None:
        if len(explicit_labels) != len(image_paths):
            raise ValueError("Số --actual-label phải bằng số ảnh --image")
        return [_parse_label(value, class_names) for value in explicit_labels]
    if manifest_path is None or not Path(manifest_path).exists():
        return [None] * len(image_paths)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = payload.get("images", {})
    labels: list[int | None] = []
    for image_path in image_paths:
        record = records.get(Path(image_path).name)
        labels.append(None if record is None else int(record["label"]))
    return labels


def plot_predictions(
    images: np.ndarray,
    results: list[dict[str, Any]],
    destination: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Vẽ ảnh đầu vào cùng predicted label, actual label và top-k score."""
    if destination is None and not show:
        raise ValueError("Cần truyền destination hoặc bật show")
    matplotlib_config = ROOT / "outputs" / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(5, len(images))
    rows = ceil(len(images) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.5 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, image, result in zip(axes.ravel(), images, results):
        actual = result.get("actual_class")
        predicted = result["predicted_class"]
        is_correct = actual is not None and result["actual_label"] == result["predicted_label"]
        color = "#187a35" if is_correct else "#b22222" if actual is not None else "#333333"
        actual_text = actual if actual is not None else "không cung cấp"
        score_text = " | ".join(
            f"{item['class']}: {item['score']:.2f}" for item in result["top_k"]
        )
        axis.imshow(image, interpolation="nearest")
        axis.set_title(
            f"Dự đoán: {predicted}\nThực tế: {actual_text}\n{score_text}",
            color=color,
            fontsize=9,
        )
    figure.suptitle("Kết quả phân loại CIFAR-10", fontsize=14)
    figure.tight_layout()
    output_path = None
    if destination is not None:
        output_path = Path(destination)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/models/hog_svm.pkl", help="Model bundle tạo bởi main.py train")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", nargs="+", help="Một hoặc nhiều đường dẫn ảnh")
    inputs.add_argument("--cifar-index", nargs="+", type=int, help="Một hoặc nhiều index trong CIFAR-10 test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--feature", choices=("pixel", "hog"), help="Override cho model bundle cũ")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--actual-label", nargs="+", help="Nhãn thật theo tên hoặc số, cùng thứ tự với --image")
    parser.add_argument("--manifest", default="demo/labels.json", help="Manifest nhãn cho ảnh demo")
    parser.add_argument(
        "--plot",
        nargs="?",
        const="demo/prediction.png",
        help="Lưu plot; nếu không ghi đường dẫn thì dùng demo/prediction.png",
    )
    parser.add_argument("--show", action="store_true", help="Mở plot để trình diễn trực tiếp")
    parser.add_argument("--output", help="Tùy chọn: lưu kết quả JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.model)

    sources: list[str]
    true_labels: list[int | None]
    if args.image:
        images = np.stack([load_image(path) for path in args.image])
        sources = [str(Path(path)) for path in args.image]
        true_labels = actual_labels_for_images(
            args.image,
            [str(name) for name in bundle["class_names"]],
            args.manifest,
            args.actual_label,
        )
    else:
        if args.actual_label:
            raise ValueError("--actual-label chỉ dùng cùng --image")
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
        result["actual_label"] = true_label
        result["actual_class"] = None if true_label is None else str(class_names[true_label])
        # Giữ hai field cũ để các JSON consumer hiện có không bị hỏng.
        result["true_label"] = true_label
        result["true_class"] = result["actual_class"]
        status = ""
        if true_label is not None:
            status = "ĐÚNG" if true_label == result["predicted_label"] else "SAI"
            status = f" | thực tế={result['actual_class']} | {status}"
        top_text = ", ".join(f"{item['class']}:{item['score']:.3f}" for item in result["top_k"])
        print(f"{source} -> dự đoán={result['predicted_class']}{status} | top-{len(result['top_k'])}: {top_text}")

    if args.plot or args.show:
        plot_path = plot_predictions(images, results, args.plot, args.show)
        if plot_path is not None:
            print(f"Đã lưu plot vào {plot_path}")

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Đã lưu kết quả vào {destination}")


if __name__ == "__main__":
    main()
