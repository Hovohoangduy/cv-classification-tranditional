"""NumPy k-means and Bag of Visual Words encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None]
        - 2 * x @ centers.T,
        0.0,
    )


@dataclass
class KMeans:
    num_clusters: int = 128
    max_iterations: int = 50
    tolerance: float = 1e-4
    seed: int = 42
    centers_: np.ndarray | None = None

    def fit(self, samples: np.ndarray) -> "KMeans":
        x = np.asarray(samples, dtype=np.float32)
        if len(x) < self.num_clusters:
            raise ValueError("Number of samples must be at least num_clusters")
        rng = np.random.default_rng(self.seed)
        centers = x[rng.choice(len(x), self.num_clusters, replace=False)].copy()
        for _ in range(self.max_iterations):
            labels = np.argmin(_squared_distances(x, centers), axis=1)
            updated = centers.copy()
            for cluster in range(self.num_clusters):
                members = x[labels == cluster]
                updated[cluster] = members.mean(axis=0) if len(members) else x[rng.integers(len(x))]
            shift = float(np.linalg.norm(updated - centers))
            centers = updated
            if shift <= self.tolerance:
                break
        self.centers_ = centers
        return self

    def predict(self, samples: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("KMeans must be fitted before prediction")
        return np.argmin(_squared_distances(np.asarray(samples, np.float32), self.centers_), axis=1)


@dataclass
class BagOfVisualWords:
    vocabulary_size: int = 128
    max_descriptors: int = 50000
    max_iterations: int = 50
    seed: int = 42
    kmeans_: KMeans | None = None

    def fit(self, descriptor_sets: list[np.ndarray]) -> "BagOfVisualWords":
        nonempty = [descriptors for descriptors in descriptor_sets if len(descriptors)]
        if not nonempty:
            raise ValueError("No descriptors were supplied")
        descriptors = np.concatenate(nonempty)
        if len(descriptors) > self.max_descriptors:
            rng = np.random.default_rng(self.seed)
            descriptors = descriptors[rng.choice(len(descriptors), self.max_descriptors, replace=False)]
        self.kmeans_ = KMeans(
            self.vocabulary_size, max_iterations=self.max_iterations, seed=self.seed
        ).fit(descriptors)
        return self

    def transform(self, descriptor_sets: list[np.ndarray]) -> np.ndarray:
        if self.kmeans_ is None:
            raise RuntimeError("BagOfVisualWords must be fitted before transform")
        output = np.zeros((len(descriptor_sets), self.vocabulary_size), dtype=np.float32)
        for index, descriptors in enumerate(descriptor_sets):
            if len(descriptors):
                words = self.kmeans_.predict(descriptors)
                output[index] = np.bincount(words, minlength=self.vocabulary_size)
                output[index] /= np.linalg.norm(output[index]) + 1e-8
        return output

    def fit_transform(self, descriptor_sets: list[np.ndarray]) -> np.ndarray:
        return self.fit(descriptor_sets).transform(descriptor_sets)
