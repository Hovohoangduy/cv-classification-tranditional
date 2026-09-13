"""Classification metrics implemented with NumPy."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    classes = num_classes or int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> dict:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    diagonal = np.diag(matrix).astype(np.float64)
    precision = np.divide(diagonal, matrix.sum(axis=0), out=np.zeros_like(diagonal), where=matrix.sum(axis=0) != 0)
    recall = np.divide(diagonal, matrix.sum(axis=1), out=np.zeros_like(diagonal), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(diagonal), where=(precision + recall) != 0)
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

