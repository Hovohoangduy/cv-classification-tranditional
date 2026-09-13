"""A compact CART classification tree with randomized split candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    def __init__(
        self,
        max_depth: int = 15,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features: int | float | str = "sqrt",
        num_thresholds: int = 16,
        seed: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.num_thresholds = num_thresholds
        self.rng = np.random.default_rng(seed)
        self.root_: Node | None = None
        self.num_classes_: int = 0

    def _feature_count(self, dimensions: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(dimensions)))
        if isinstance(self.max_features, float):
            return max(1, min(dimensions, int(round(dimensions * self.max_features))))
        return max(1, min(dimensions, int(self.max_features)))

    def _gini(self, labels: np.ndarray) -> float:
        counts = np.bincount(labels, minlength=self.num_classes_).astype(np.float64)
        probabilities = counts / max(1, len(labels))
        return float(1.0 - np.dot(probabilities, probabilities))

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        feature_ids = self.rng.choice(x.shape[1], self._feature_count(x.shape[1]), replace=False)
        best_score, best_feature, best_threshold = np.inf, None, None
        for feature in feature_ids:
            values = x[:, feature]
            low, high = float(values.min()), float(values.max())
            if low == high:
                continue
            unique = np.unique(values)
            if len(unique) <= self.num_thresholds + 1:
                thresholds = (unique[:-1] + unique[1:]) * 0.5
            else:
                thresholds = self.rng.uniform(low, high, self.num_thresholds)
            for threshold in thresholds:
                mask = values <= threshold
                left_count = int(mask.sum())
                right_count = len(y) - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                score = (left_count * self._gini(y[mask]) + right_count * self._gini(y[~mask])) / len(y)
                if score < best_score:
                    best_score, best_feature, best_threshold = score, int(feature), float(threshold)
        return best_feature, best_threshold

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        prediction = int(np.argmax(np.bincount(y, minlength=self.num_classes_)))
        node = Node(prediction=prediction)
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return node
        feature, threshold = self._best_split(x, y)
        if feature is None or threshold is None:
            return node
        mask = x[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(x[mask], y[mask], depth + 1)
        node.right = self._grow(x[~mask], y[~mask], depth + 1)
        return node

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "DecisionTreeClassifier":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        self.num_classes_ = int(y.max()) + 1
        self.root_ = self._grow(x, y, 0)
        return self

    def _predict_one(self, sample: np.ndarray) -> int:
        if self.root_ is None:
            raise RuntimeError("Tree must be fitted before prediction")
        node = self.root_
        while not node.is_leaf:
            node = node.left if sample[node.feature] <= node.threshold else node.right  # type: ignore[index]
            if node is None:
                raise RuntimeError("Malformed decision tree")
        return node.prediction

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(sample) for sample in features], dtype=np.int64)

