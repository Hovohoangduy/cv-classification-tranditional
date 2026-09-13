"""Random Forest built on the local CART implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decision_tree import DecisionTreeClassifier


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
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        self.classes_ = np.unique(y)
        expected = np.arange(len(self.classes_))
        if not np.array_equal(self.classes_, expected):
            raise ValueError("RandomForest labels must be contiguous integers starting at zero")
        rng = np.random.default_rng(self.seed)
        sample_count = max(2, int(round(len(x) * self.sample_fraction)))
        self.trees_ = []
        for index in range(self.num_trees):
            bootstrap = rng.integers(0, len(x), size=sample_count)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                num_thresholds=self.num_thresholds,
                seed=self.seed + index + 1,
            )
            tree.fit(x[bootstrap], y[bootstrap])
            self.trees_.append(tree)
            if self.verbose:
                print(f"tree={index + 1:03d}/{self.num_trees}")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.trees_ or self.classes_ is None:
            raise RuntimeError("Forest must be fitted before prediction")
        votes = np.stack([tree.predict(features) for tree in self.trees_])
        predictions = np.empty(votes.shape[1], dtype=np.int64)
        for index in range(votes.shape[1]):
            predictions[index] = np.argmax(np.bincount(votes[:, index], minlength=len(self.classes_)))
        return predictions

