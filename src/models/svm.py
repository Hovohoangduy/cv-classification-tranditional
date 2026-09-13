"""Multiclass linear SVM trained with mini-batch SGD."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearSVM:
    learning_rate: float = 0.01
    regularization: float = 1e-4
    epochs: int = 30
    batch_size: int = 256
    seed: int = 42
    verbose: bool = False
    weights_: np.ndarray | None = None
    bias_: np.ndarray | None = None
    classes_: np.ndarray | None = None
    loss_history_: list[float] = field(default_factory=list)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LinearSVM":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected X=(n_samples, n_features) and y=(n_samples,)")
        self.classes_ = np.unique(y)
        class_to_index = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([class_to_index[label] for label in y], dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(0, 0.001, (len(self.classes_), x.shape[1])).astype(np.float32)
        self.bias_ = np.zeros(len(self.classes_), dtype=np.float32)
        self.loss_history_ = []

        for epoch in range(self.epochs):
            order = rng.permutation(len(x))
            learning_rate = self.learning_rate / np.sqrt(epoch + 1)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(x), self.batch_size):
                indices = order[start : start + self.batch_size]
                xb = x[indices]
                target = encoded[indices]
                scores = xb @ self.weights_.T + self.bias_
                correct = scores[np.arange(len(xb)), target]
                margins = np.maximum(0.0, scores - correct[:, None] + 1.0)
                margins[np.arange(len(xb)), target] = 0.0
                active = (margins > 0).astype(np.float32)
                active[np.arange(len(xb)), target] = -active.sum(axis=1)
                gradient_w = active.T @ xb / len(xb) + self.regularization * self.weights_
                gradient_b = active.mean(axis=0)
                self.weights_ -= learning_rate * gradient_w
                self.bias_ -= learning_rate * gradient_b
                epoch_loss += float(margins.sum() / len(xb) + 0.5 * self.regularization * np.sum(self.weights_**2))
                batches += 1
            self.loss_history_.append(epoch_loss / max(1, batches))
            if self.verbose:
                print(f"epoch={epoch + 1:03d} loss={self.loss_history_[-1]:.6f}")
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        return np.asarray(features, dtype=np.float32) @ self.weights_.T + self.bias_

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.classes_[np.argmax(self.decision_function(features), axis=1)]

