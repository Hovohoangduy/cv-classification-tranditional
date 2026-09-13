"""Small reproducibility and persistence helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def save_json(data: dict[str, Any], path: str | Path) -> None:
    destination = ensure_parent(path)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_pickle(value: Any, path: str | Path) -> None:
    destination = ensure_parent(path)
    with destination.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def limit_stratified(indices: np.ndarray, labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    indexed_labels = labels[indices]
    for label in np.unique(indexed_labels):
        candidates = indices[indexed_labels == label]
        count = max(1, int(round(limit * len(candidates) / len(indices))))
        selected.append(rng.choice(candidates, min(count, len(candidates)), replace=False))
    result = np.concatenate(selected)
    if len(result) > limit:
        result = rng.choice(result, limit, replace=False)
    rng.shuffle(result)
    return result

