#!/usr/bin/env python3
"""Train a classifier from a cached feature archive and evaluate it once."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import classification_metrics
from src.models import LinearSVM, RandomForestClassifier
from src.preprocessing import StandardScaler
from src.utils import load_json, save_json, save_pickle


def make_model(config: dict):
    name = config["model"]
    parameters = config.get("model_parameters", {})
    if name == "linear_svm":
        return LinearSVM(**parameters)
    if name == "random_forest":
        return RandomForestClassifier(**parameters)
    raise ValueError(f"Unknown model: {name}")


def plot_confusion(matrix: list[list[int]], labels: np.ndarray, path: Path) -> None:
    values = np.asarray(matrix)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Nhãn dự đoán")
    axis.set_ylabel("Nhãn thực")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_training_loss(history: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(np.arange(1, len(history) + 1), history, marker="o", markersize=3)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Hinge loss")
    axis.set_title("Quá trình hội tụ trên tập train")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_class(metrics: dict, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(labels))
    width = 0.26
    fig, axis = plt.subplots(figsize=(8, 4.2))
    axis.bar(positions - width, metrics["per_class_precision"], width, label="Precision")
    axis.bar(positions, metrics["per_class_recall"], width, label="Recall")
    axis.bar(positions + width, metrics["per_class_f1"], width, label="F1")
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Giá trị")
    axis.set_title("Kết quả theo từng lớp trên tập test")
    axis.legend(ncols=3)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    archive = np.load(config["features"], allow_pickle=False)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(archive["x_train"])
    x_validation = scaler.transform(archive["x_validation"])
    x_test = scaler.transform(archive["x_test"])
    model = make_model(config)

    started = time.perf_counter()
    model.fit(x_train, archive["y_train"])
    training_seconds = time.perf_counter() - started
    validation_started = time.perf_counter()
    validation_prediction = model.predict(x_validation)
    test_prediction = model.predict(x_test)
    test_scores = model.decision_function(x_test) if hasattr(model, "decision_function") else np.empty((0, 0))
    inference_seconds = time.perf_counter() - validation_started
    validation_metrics = classification_metrics(archive["y_validation"], validation_prediction, len(archive["class_names"]))
    test_metrics = classification_metrics(archive["y_test"], test_prediction, len(archive["class_names"]))

    result = {
        "experiment": config.get("name", Path(args.config).stem),
        "feature": str(archive["feature_name"]),
        "model": config["model"],
        "model_parameters": config.get("model_parameters", {}),
        "seed": config.get("model_parameters", {}).get("seed", 42),
        "samples": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
        "feature_dimensions": x_train.shape[1],
        "extraction_seconds": float(archive["extraction_seconds"]),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "validation": validation_metrics,
        "test": test_metrics,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    metrics_path = Path(config["metrics_output"])
    model_path = Path(config["model_output"])
    figure_path = Path(config["confusion_output"])
    result["artifacts"] = {
        key: config[key]
        for key in ("model_output", "confusion_output", "loss_output", "per_class_output", "predictions_output")
        if key in config
    }
    save_json(result, metrics_path)
    save_pickle(
        {
            "model": model,
            "scaler": scaler,
            "class_names": archive["class_names"],
            "feature_name": str(archive["feature_name"]),
            "feature_parameters": config.get("feature_parameters", {}),
        },
        model_path,
    )
    plot_confusion(test_metrics["confusion_matrix"], archive["class_names"], figure_path)
    if "loss_output" in config and getattr(model, "loss_history_", None):
        plot_training_loss(model.loss_history_, Path(config["loss_output"]))
    if "per_class_output" in config:
        plot_per_class(test_metrics, archive["class_names"], Path(config["per_class_output"]))
    if "predictions_output" in config:
        prediction_path = Path(config["predictions_output"])
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if "test_indices" in archive.files:
            test_indices = archive["test_indices"]
        elif len(test_prediction) == 10000:
            test_indices = np.arange(10000)
        else:
            test_indices = np.full(len(test_prediction), -1, dtype=np.int64)
        np.savez_compressed(
            prediction_path,
            test_indices=test_indices,
            y_true=archive["y_test"],
            y_pred=test_prediction,
            scores=test_scores,
            class_names=archive["class_names"],
        )
    print(f"validation accuracy={validation_metrics['accuracy']:.4f}, macro-F1={validation_metrics['macro_f1']:.4f}")
    print(f"test accuracy={test_metrics['accuracy']:.4f}, macro-F1={test_metrics['macro_f1']:.4f}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
