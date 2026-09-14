"""Linear SVM, CART và Random Forest cài đặt tối giản bằng NumPy."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import StandardScaler


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
        x, y = np.asarray(features, dtype=np.float32), np.asarray(labels)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("X phải có shape (N, D), y phải có shape (N,)")
        self.classes_ = np.unique(y)
        mapping = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([mapping[label] for label in y], dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(0, 0.001, (len(self.classes_), x.shape[1])).astype(np.float32)
        self.bias_ = np.zeros(len(self.classes_), dtype=np.float32)
        self.loss_history_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(len(x))
            rate = self.learning_rate / np.sqrt(epoch + 1)
            total, batches = 0.0, 0
            for start in range(0, len(x), self.batch_size):
                ids = order[start : start + self.batch_size]
                batch, target = x[ids], encoded[ids]
                scores = batch @ self.weights_.T + self.bias_
                correct = scores[np.arange(len(batch)), target]
                margins = np.maximum(0.0, scores - correct[:, None] + 1.0)
                margins[np.arange(len(batch)), target] = 0.0
                active = (margins > 0).astype(np.float32)
                active[np.arange(len(batch)), target] = -active.sum(axis=1)
                gradient_w = active.T @ batch / len(batch) + self.regularization * self.weights_
                self.weights_ -= rate * gradient_w
                self.bias_ -= rate * active.mean(axis=0)
                total += float(margins.sum() / len(batch) + 0.5 * self.regularization * np.sum(self.weights_**2))
                batches += 1
            self.loss_history_.append(total / max(1, batches))
            if self.verbose:
                print(f"epoch={epoch + 1:03d} loss={self.loss_history_[-1]:.6f}")
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("SVM chưa được fit")
        return np.asarray(features, dtype=np.float32) @ self.weights_.T + self.bias_

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("SVM chưa được fit")
        return self.classes_[np.argmax(self.decision_function(features), axis=1)]


@dataclass
class Node:
    prediction: int
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 15, min_samples_split: int = 4,
                 min_samples_leaf: int = 2, max_features: int | float | str = "sqrt",
                 num_thresholds: int = 16, seed: int = 42) -> None:
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.min_samples_leaf, self.max_features = min_samples_leaf, max_features
        self.num_thresholds = num_thresholds
        self.rng = np.random.default_rng(seed)
        self.root_: Node | None = None
        self.num_classes_ = 0

    def _feature_count(self, dimensions: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(dimensions)))
        if isinstance(self.max_features, float):
            return max(1, min(dimensions, int(round(dimensions * self.max_features))))
        return max(1, min(dimensions, int(self.max_features)))

    def _gini(self, labels: np.ndarray) -> float:
        probability = np.bincount(labels, minlength=self.num_classes_) / max(1, len(labels))
        return float(1.0 - np.dot(probability, probability))

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        candidates = self.rng.choice(x.shape[1], self._feature_count(x.shape[1]), replace=False)
        best_score, best_feature, best_threshold = np.inf, None, None
        for feature in candidates:
            values = x[:, feature]
            low, high = float(values.min()), float(values.max())
            if low == high:
                continue
            unique = np.unique(values)
            thresholds = ((unique[:-1] + unique[1:]) * 0.5 if len(unique) <= self.num_thresholds + 1
                          else self.rng.uniform(low, high, self.num_thresholds))
            for threshold in thresholds:
                mask = values <= threshold
                left_count, right_count = int(mask.sum()), int((~mask).sum())
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                score = (left_count * self._gini(y[mask]) + right_count * self._gini(y[~mask])) / len(y)
                if score < best_score:
                    best_score, best_feature, best_threshold = score, int(feature), float(threshold)
        return best_feature, best_threshold

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(int(np.argmax(np.bincount(y, minlength=self.num_classes_))))
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return node
        feature, threshold = self._best_split(x, y)
        if feature is None:
            return node
        mask = x[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left, node.right = self._grow(x[mask], y[mask], depth + 1), self._grow(x[~mask], y[~mask], depth + 1)
        return node

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "DecisionTreeClassifier":
        x, y = np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)
        self.num_classes_ = int(y.max()) + 1
        self.root_ = self._grow(x, y, 0)
        return self

    def _predict_one(self, sample: np.ndarray) -> int:
        if self.root_ is None:
            raise RuntimeError("Decision tree chưa được fit")
        node = self.root_
        while not node.is_leaf:
            node = node.left if sample[node.feature] <= node.threshold else node.right  # type: ignore[index]
            if node is None:
                raise RuntimeError("Cấu trúc cây không hợp lệ")
        return node.prediction

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(sample) for sample in features], dtype=np.int64)


@dataclass
class RandomForestClassifier:
    num_trees: int = 50
    max_depth: int = 15
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    max_features: int | float | str = "sqrt"
    num_thresholds: int = 16
    sample_fraction: float = 1.0
    seed: int = 42
    verbose: bool = False
    trees_: list[DecisionTreeClassifier] = field(default_factory=list)
    classes_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "RandomForestClassifier":
        x, y = np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            raise ValueError("Nhãn Random Forest phải là các số nguyên liên tiếp từ 0")
        rng, count = np.random.default_rng(self.seed), max(2, int(round(len(x) * self.sample_fraction)))
        self.trees_ = []
        for index in range(self.num_trees):
            bootstrap = rng.integers(0, len(x), size=count)
            tree = DecisionTreeClassifier(self.max_depth, self.min_samples_split, self.min_samples_leaf,
                                          self.max_features, self.num_thresholds, self.seed + index + 1)
            self.trees_.append(tree.fit(x[bootstrap], y[bootstrap]))
            if self.verbose:
                print(f"tree={index + 1:03d}/{self.num_trees}")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.trees_ or self.classes_ is None:
            raise RuntimeError("Random Forest chưa được fit")
        votes = np.stack([tree.predict(features) for tree in self.trees_])
        return np.asarray([np.argmax(np.bincount(votes[:, i], minlength=len(self.classes_)))
                           for i in range(votes.shape[1])], dtype=np.int64)


class _CompatibleUnpickler(pickle.Unpickler):
    """Đọc model cũ dù cấu trúc package đã được tinh gọn."""

    MAPPING: dict[tuple[str, str], type] = {
        ("src.preprocessing", "StandardScaler"): StandardScaler,
        ("src.models.svm", "LinearSVM"): LinearSVM,
        ("src.models.decision_tree", "Node"): Node,
        ("src.models.decision_tree", "DecisionTreeClassifier"): DecisionTreeClassifier,
        ("src.models.random_forest", "RandomForestClassifier"): RandomForestClassifier,
        ("cv.data", "StandardScaler"): StandardScaler,
        ("cv.models", "LinearSVM"): LinearSVM,
        ("cv.models", "Node"): Node,
        ("cv.models", "DecisionTreeClassifier"): DecisionTreeClassifier,
        ("cv.models", "RandomForestClassifier"): RandomForestClassifier,
    }

    def find_class(self, module: str, name: str) -> type:
        mapped = self.MAPPING.get((module, name))
        return mapped if mapped is not None else super().find_class(module, name)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Nạp model bundle do project tạo ra; không dùng với pickle không tin cậy."""
    with Path(path).open("rb") as stream:
        return _CompatibleUnpickler(stream).load()
