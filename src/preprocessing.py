"""Preprocessing functions fitted exclusively on training data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def to_float(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    return images.astype(np.float32) / (255.0 if images.dtype == np.uint8 else 1.0)


def rgb_to_gray(images: np.ndarray) -> np.ndarray:
    images = to_float(images)
    if images.shape[-1] != 3:
        raise ValueError("Expected RGB images with channels in the last dimension")
    return np.tensordot(images, np.array([0.299, 0.587, 0.114], np.float32), axes=([-1], [0]))


@dataclass
class StandardScaler:
    epsilon: float = 1e-8
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "StandardScaler":
        values = np.asarray(features, dtype=np.float32)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < self.epsilon] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        return (np.asarray(features, dtype=np.float32) - self.mean_) / self.scale_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

