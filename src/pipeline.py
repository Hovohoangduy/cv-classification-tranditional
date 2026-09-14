"""Luồng thí nghiệm dùng chung cho CLI: extract, train, evaluate và evidence."""

from __future__ import annotations

import json
import os
import pickle
import platform
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import StandardScaler, limit_stratified, load_cifar10, stratified_split, to_float
from .features import BagOfVisualWords, HOG, SIFT
from .models import LinearSVM, RandomForestClassifier, load_bundle


def save_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    diagonal = np.diag(matrix).astype(np.float64)
    precision = np.divide(diagonal, matrix.sum(axis=0), out=np.zeros(num_classes), where=matrix.sum(axis=0) != 0)
    recall = np.divide(diagonal, matrix.sum(axis=1), out=np.zeros(num_classes), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes), where=precision + recall != 0)
    return {
        "accuracy": float(diagonal.sum() / max(1, matrix.sum())),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


import multiprocessing

def _process_one_sift(args):
    image, extractor = args
    return extractor.detect_and_describe(image)[1]

def _sift_descriptors(images: np.ndarray, extractor: SIFT, split: str) -> list[np.ndarray]:
    cache_path = Path(f"outputs/features/raw_sift_{split}_{len(images)}.pkl")
    if cache_path.exists():
        print(f"Loading cached SIFT cho {split} ({len(images)} ảnh)...")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    print(f"Bắt đầu trích xuất SIFT cho {split} ({len(images)} ảnh)...")
    output = []
    args = [(img, extractor) for img in images]
    with multiprocessing.Pool() as pool:
        for i, res in enumerate(pool.imap(_process_one_sift, args, chunksize=50)):
            output.append(res)
            if (i + 1) % 500 == 0:
                print(f"SIFT {split}: {i + 1}/{len(images)}")
    
    with cache_path.open("wb") as f:
        pickle.dump(output, f)
    return output


def extract_archive(feature: str, output: str | Path, data_dir: str | Path = "data", seed: int = 42,
                    train_limit: int | None = None, test_limit: int | None = None,
                    cell_size: int = 4, block_size: int = 2, num_bins: int = 9,
                    vocabulary_size: int = 128) -> Path:
    dataset = load_cifar10(data_dir)
    train_ids, validation_ids = stratified_split(dataset.train_labels, seed=seed)
    train_ids = limit_stratified(train_ids, dataset.train_labels, train_limit, seed)
    validation_limit = None if train_limit is None else max(10, train_limit // 9)
    validation_ids = limit_stratified(validation_ids, dataset.train_labels, validation_limit, seed + 1)
    test_ids = limit_stratified(np.arange(len(dataset.test_labels)), dataset.test_labels, test_limit, seed + 2)
    image_sets = (dataset.train_images[train_ids], dataset.train_images[validation_ids], dataset.test_images[test_ids])
    started = time.perf_counter()
    if feature == "pixel":
        feature_sets = tuple(to_float(images).reshape(len(images), -1) for images in image_sets)
    elif feature == "hog":
        extractor = HOG(cell_size=cell_size, block_size=block_size, num_bins=num_bins)
        feature_sets = tuple(extractor.transform(images, verbose=True) for images in image_sets)
    elif feature == "sift":
        sift = SIFT()
        descriptors = tuple(_sift_descriptors(images, sift, split) for images, split in zip(image_sets, ("train", "validation", "test")))
        def pad_and_flatten(desc_list, max_kp, dim):
            out = []
            for d in desc_list:
                if len(d) == 0:
                    out.append(np.zeros(max_kp * dim, dtype=np.float32))
                else:
                    flat = d.flatten()
                    if len(flat) < max_kp * dim:
                        flat = np.pad(flat, (0, max_kp * dim - len(flat)))
                    else:
                        flat = flat[:max_kp * dim]
                    out.append(flat)
            return np.array(out)
        feature_sets = tuple(pad_and_flatten(items, sift.max_keypoints, 128) for items in descriptors)
    elif feature == "sift_bovw":
        sift = SIFT()
        descriptors = tuple(_sift_descriptors(images, sift, split) for images, split in zip(image_sets, ("train", "validation", "test")))
        vocabulary = BagOfVisualWords(vocabulary_size=vocabulary_size).fit(descriptors[0])
        feature_sets = tuple(vocabulary.transform(items) for items in descriptors)
    else:
        raise ValueError(f"Feature không hợp lệ: {feature}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, x_train=feature_sets[0], y_train=dataset.train_labels[train_ids],
        x_validation=feature_sets[1], y_validation=dataset.train_labels[validation_ids],
        x_test=feature_sets[2], y_test=dataset.test_labels[test_ids],
        train_indices=train_ids, validation_indices=validation_ids, test_indices=test_ids,
        class_names=np.asarray(dataset.class_names), extraction_seconds=np.asarray(time.perf_counter() - started),
        feature_name=np.asarray(feature),
    )
    print(f"Đã lưu {feature} {feature_sets[0].shape} vào {destination}")
    return destination


def _plot_confusion(matrix: list[list[int]], labels: np.ndarray, 
                    path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(np.asarray(matrix), cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Nhãn dự đoán")
    axis.set_ylabel("Nhãn thực")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_loss(history: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(np.arange(1, len(history) + 1), history, marker="o", markersize=3)
    axis.set(xlabel="Epoch", ylabel="Hinge loss", title="Quá trình hội tụ trên tập train")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_per_class(metrics: dict[str, Any], labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions, width = np.arange(len(labels)), 0.26
    fig, axis = plt.subplots(figsize=(8, 4.2))
    axis.bar(positions - width, metrics["per_class_precision"], width, label="Precision")
    axis.bar(positions, metrics["per_class_recall"], width, label="Recall")
    axis.bar(positions + width, metrics["per_class_f1"], width, label="F1")
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set(ylim=(0, 1), ylabel="Giá trị", title="Kết quả theo từng lớp trên tập test")
    axis.legend(ncols=3)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_experiment(features_path: str | Path, name: str, model_name: str = "svm",
                     model_parameters: dict[str, Any] | None = None,
                     feature_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    archive = np.load(features_path, allow_pickle=False)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(archive["x_train"])
    x_validation, x_test = scaler.transform(archive["x_validation"]), scaler.transform(archive["x_test"])
    parameters = model_parameters or {}
    model = LinearSVM(**parameters) if model_name == "svm" else RandomForestClassifier(**parameters)
    started = time.perf_counter()
    model.fit(x_train, archive["y_train"])
    training_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    validation_prediction, test_prediction = model.predict(x_validation), model.predict(x_test)
    scores = model.decision_function(x_test) if hasattr(model, "decision_function") else np.empty((0, 0))
    inference_seconds = time.perf_counter() - inference_started
    labels = archive["class_names"]
    validation_metrics = classification_metrics(archive["y_validation"], validation_prediction, len(labels))
    test_metrics = classification_metrics(archive["y_test"], test_prediction, len(labels))
    model_label = "linear_svm" if model_name == "svm" else "random_forest"
    paths = {
        "model_output": f"outputs/models/{name}.pkl",
        "confusion_output": f"outputs/figures/{name}_confusion.png",
        "loss_output": f"outputs/figures/{name}_loss.png",
        "per_class_output": f"outputs/figures/{name}_per_class.png",
        "predictions_output": f"outputs/predictions/{name}.npz",
    }
    result = {
        "experiment": name, "feature": str(archive["feature_name"]), "model": model_label,
        "model_parameters": parameters, "seed": parameters.get("seed", 42),
        "samples": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
        "feature_dimensions": x_train.shape[1], "extraction_seconds": float(archive["extraction_seconds"]),
        "training_seconds": training_seconds, "inference_seconds": inference_seconds,
        "validation": validation_metrics, "test": test_metrics,
        "environment": {"python": platform.python_version(), "numpy": np.__version__}, "artifacts": paths,
    }
    save_json(result, f"outputs/metrics/{name}.json")
    Path(paths["model_output"]).parent.mkdir(parents=True, exist_ok=True)
    with Path(paths["model_output"]).open("wb") as stream:
        pickle.dump({"model": model, "scaler": scaler, "class_names": labels,
                     "feature_name": str(archive["feature_name"]),
                     "feature_parameters": feature_parameters or {}}, stream)
    _plot_confusion(test_metrics["confusion_matrix"], labels, Path(paths["confusion_output"]))
    if getattr(model, "loss_history_", None):
        _plot_loss(model.loss_history_, Path(paths["loss_output"]))
    _plot_per_class(test_metrics, labels, Path(paths["per_class_output"]))
    test_indices = archive["test_indices"] if "test_indices" in archive.files else np.full(len(test_prediction), -1)
    Path(paths["predictions_output"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["predictions_output"], test_indices=test_indices, y_true=archive["y_test"],
                        y_pred=test_prediction, scores=scores, class_names=labels)
    print(f"validation accuracy={validation_metrics['accuracy']:.4f}; test accuracy={test_metrics['accuracy']:.4f}")
    return result


def evaluate(model_path: str | Path, features_path: str | Path, split: str = "test") -> dict[str, Any]:
    bundle, archive = load_bundle(model_path), np.load(features_path, allow_pickle=False)
    predictions = bundle["model"].predict(bundle["scaler"].transform(archive[f"x_{split}"]))
    return classification_metrics(archive[f"y_{split}"], predictions, len(bundle["class_names"]))


def generate_evidence(predictions_path: str | Path, name: str, data_dir: str | Path = "data") -> dict[str, Any]:
    prediction = np.load(predictions_path, allow_pickle=False)
    test_ids, y_true, y_pred = prediction["test_indices"].astype(int), prediction["y_true"].astype(int), prediction["y_pred"].astype(int)
    if np.any(test_ids < 0):
        raise ValueError("Artifact prediction không chứa CIFAR-10 test index hợp lệ")
    scores, names = prediction["scores"], prediction["class_names"]
    values = (np.partition(scores, -2, axis=1)[:, -1] - np.partition(scores, -2, axis=1)[:, -2]
              if scores.ndim == 2 and scores.shape[1] >= 2 else np.zeros(len(y_true)))
    correct = y_true == y_pred
    selected = []
    for is_correct in (True, False):
        for label in np.unique(y_true):
            candidates = np.flatnonzero((y_true == label) & (correct == is_correct))
            if len(candidates):
                selected.append(int(candidates[np.argmax(values[candidates])]))
    images = load_cifar10(data_dir, download=False).test_images[test_ids]
    gallery_path = Path(f"outputs/figures/{name}_predictions.png")
    rows, columns = int(np.ceil(len(selected) / 5)), 5
    fig, axes = plt.subplots(rows, columns, figsize=(11, 2.25 * rows), squeeze=False)
    for axis in axes.ravel(): axis.axis("off")
    for axis, position in zip(axes.ravel(), selected):
        color = "#167c3a" if correct[position] else "#b22222"
        axis.imshow(images[position]); axis.axis("off")
        axis.set_title(f"Thật: {names[y_true[position]]}\nĐoán: {names[y_pred[position]]}\nmargin={values[position]:.2f}", fontsize=8, color=color)
    fig.suptitle("Ví dụ dự đoán có decision margin cao theo từng lớp")
    fig.tight_layout(); gallery_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(gallery_path, dpi=200); plt.close(fig)
    margin_path = Path(f"outputs/figures/{name}_margin.png")
    fig, axis = plt.subplots(figsize=(6.4, 3.8)); axis.hist(values[correct], bins=35, alpha=.65, density=True, label="Đúng")
    axis.hist(values[~correct], bins=35, alpha=.65, density=True, label="Sai"); axis.legend(); axis.grid(alpha=.2)
    axis.set(xlabel="Decision margin", ylabel="Mật độ", title="Phân bố độ chắc chắn"); fig.tight_layout(); fig.savefig(margin_path, dpi=180); plt.close(fig)
    matrix = confusion_matrix(y_true, y_pred, len(names)); off = matrix.copy(); np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:10]
    summary = {"num_samples": len(y_true), "num_correct": int(correct.sum()), "num_errors": int((~correct).sum()),
               "accuracy": float(correct.mean()), "mean_margin_correct": float(values[correct].mean()),
               "mean_margin_error": float(values[~correct].mean()),
               "median_margin_correct": float(np.median(values[correct])),
               "median_margin_error": float(np.median(values[~correct])),
               "top_confusion_pairs": [{"true": str(names[i]), "predicted": str(names[j]), "count": int(off[i, j])} for i, j in pairs],
               "gallery_indices": [int(test_ids[index]) for index in selected]}
    save_json(summary, f"outputs/evidence/{name}.json")
    return summary


def generate_report_table(metric_paths: list[str], output: str | Path = "report/assets/results/generated_results.tex") -> None:
    rows = []
    for path in metric_paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        escape = lambda value: str(value).replace("_", r"\_").replace("%", r"\%")
        rows.append("{} & {} & {} & {:.4f} & {:.4f}".format(
            escape(result["feature"]), escape(result["model"]), result["samples"]["test"],
            result["test"]["accuracy"], result["test"]["macro_f1"]))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(" \\\\\n".join(rows), encoding="utf-8")
