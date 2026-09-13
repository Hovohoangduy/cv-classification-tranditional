#!/usr/bin/env python3
"""Create prediction galleries and quantitative evidence from saved predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_cifar10
from src.metrics import confusion_matrix
from src.utils import save_json


def margins(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] < 2:
        return np.zeros(len(scores), dtype=np.float32)
    partitioned = np.partition(scores, -2, axis=1)
    return partitioned[:, -1] - partitioned[:, -2]


def pick_by_class(y_true: np.ndarray, y_pred: np.ndarray, values: np.ndarray) -> list[int]:
    selected: list[int] = []
    classes = np.unique(y_true)
    for correct in (True, False):
        for label in classes:
            candidates = np.flatnonzero((y_true == label) & ((y_true == y_pred) == correct))
            if len(candidates):
                selected.append(int(candidates[np.argmax(values[candidates])]))
    return selected


def plot_gallery(
    images: np.ndarray,
    indices: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    values: np.ndarray,
    names: np.ndarray,
    output: Path,
) -> None:
    columns = 5
    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 2.25 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, position in zip(axes.ravel(), indices):
        true_name = names[y_true[position]]
        predicted_name = names[y_pred[position]]
        color = "#167c3a" if y_true[position] == y_pred[position] else "#b22222"
        axis.imshow(images[position])
        axis.set_title(f"Thật: {true_name}\nĐoán: {predicted_name}\nmargin={values[position]:.2f}", fontsize=8, color=color)
        axis.axis("off")
    fig.suptitle("Ví dụ dự đoán có decision margin cao theo từng lớp", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_margin_distribution(values: np.ndarray, correct: np.ndarray, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.hist(values[correct], bins=35, alpha=0.65, density=True, label="Dự đoán đúng")
    axis.hist(values[~correct], bins=35, alpha=0.65, density=True, label="Dự đoán sai")
    axis.set_xlabel("Decision margin (top-1 trừ top-2)")
    axis.set_ylabel("Mật độ")
    axis.set_title("Phân bố độ chắc chắn của mô hình")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--gallery-output", required=True)
    parser.add_argument("--margin-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    prediction = np.load(args.predictions, allow_pickle=False)
    test_indices = prediction["test_indices"].astype(np.int64)
    if np.any(test_indices < 0):
        raise ValueError("Prediction artifact does not contain valid CIFAR-10 test indices")
    y_true = prediction["y_true"].astype(np.int64)
    y_pred = prediction["y_pred"].astype(np.int64)
    scores = prediction["scores"]
    names = prediction["class_names"]
    values = margins(scores)
    correct = y_true == y_pred
    dataset = load_cifar10(args.data_dir, download=False)
    images = dataset.test_images[test_indices]

    chosen = pick_by_class(y_true, y_pred, values)
    plot_gallery(images, chosen, y_true, y_pred, values, names, Path(args.gallery_output))
    plot_margin_distribution(values, correct, Path(args.margin_output))

    matrix = confusion_matrix(y_true, y_pred, len(names))
    without_diagonal = matrix.copy()
    np.fill_diagonal(without_diagonal, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(without_diagonal.ravel())[::-1], without_diagonal.shape))[0]
    top_pairs = [
        {"true": str(names[i]), "predicted": str(names[j]), "count": int(without_diagonal[i, j])}
        for i, j in pairs[:10]
    ]
    summary = {
        "num_samples": len(y_true),
        "num_correct": int(correct.sum()),
        "num_errors": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "mean_margin_correct": float(values[correct].mean()),
        "mean_margin_error": float(values[~correct].mean()),
        "median_margin_correct": float(np.median(values[correct])),
        "median_margin_error": float(np.median(values[~correct])),
        "top_confusion_pairs": top_pairs,
        "gallery_indices": [int(test_indices[index]) for index in chosen],
    }
    save_json(summary, args.summary_output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

