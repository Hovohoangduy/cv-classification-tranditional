"""Tải CIFAR-10, chia dữ liệu và tiền xử lý."""

from __future__ import annotations

import hashlib
import json
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


@dataclass(frozen=True)
class CIFAR10:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    class_names: tuple[str, ...]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_cifar10(root: str | Path = "data") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-10-batches-py"
    archive = root / "cifar-10-python.tar.gz"
    if extracted.exists():
        return extracted
    if not archive.exists() or _md5(archive) != CIFAR10_MD5:
        print(f"Đang tải CIFAR-10 từ {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, archive)
    if _md5(archive) != CIFAR10_MD5:
        raise RuntimeError("MD5 của CIFAR-10 không khớp bản chính thức")
    root_resolved = root.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise RuntimeError("Archive chứa đường dẫn không an toàn")
        tar.extractall(root, members=members)
    return extracted


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def load_cifar10(root: str | Path = "data", download: bool = True) -> CIFAR10:
    folder = Path(root) / "cifar-10-batches-py"
    if not folder.exists():
        if not download:
            raise FileNotFoundError(f"Không tìm thấy CIFAR-10 trong {folder}")
        folder = download_cifar10(root)
    parts = [_read_batch(folder / f"data_batch_{index}") for index in range(1, 6)]
    train_images = np.concatenate([part[0] for part in parts])
    train_labels = np.concatenate([part[1] for part in parts])
    test_images, test_labels = _read_batch(folder / "test_batch")
    with (folder / "batches.meta").open("rb") as stream:
        meta = pickle.load(stream, encoding="bytes")
    class_names = tuple(name.decode("utf-8") for name in meta[b"label_names"])
    return CIFAR10(train_images, train_labels, test_images, test_labels, class_names)


def export_demo_images(
    root: str | Path = "data",
    destination: str | Path = "demo/images",
    manifest: str | Path = "demo/labels.json",
    samples_per_class: int = 1,
    seed: int = 42,
) -> Path:
    """Xuất ảnh test CIFAR-10 có nhãn thật để dùng cho live demo."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class phải lớn hơn hoặc bằng 1")
    from PIL import Image

    dataset = load_cifar10(root, download=True)
    output_dir = Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: dict[str, dict[str, int | str]] = {}
    for label, class_name in enumerate(dataset.class_names):
        candidates = np.flatnonzero(dataset.test_labels == label)
        if samples_per_class > len(candidates):
            raise ValueError(f"Lớp {class_name} chỉ có {len(candidates)} ảnh test")
        selected = np.sort(rng.choice(candidates, samples_per_class, replace=False))
        for test_index in selected:
            filename = f"{class_name}_{int(test_index):05d}.png"
            Image.fromarray(dataset.test_images[test_index]).save(output_dir / filename)
            records[filename] = {
                "label": label,
                "class_name": class_name,
                "test_index": int(test_index),
            }
    manifest_path = Path(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "CIFAR-10",
                "split": "test",
                "class_names": list(dataset.class_names),
                "images": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def stratified_split(
    labels: np.ndarray, validation_fraction: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction phải nằm trong (0, 1)")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_parts, validation_parts = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, round(len(indices) * validation_fraction))
        validation_parts.append(indices[:count])
        train_parts.append(indices[count:])
    train = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def limit_stratified(indices: np.ndarray, labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    indexed_labels = labels[indices]
    selected = []
    for label in np.unique(indexed_labels):
        candidates = indices[indexed_labels == label]
        count = max(1, round(limit * len(candidates) / len(indices)))
        selected.append(rng.choice(candidates, min(count, len(candidates)), replace=False))
    result = np.concatenate(selected)
    if len(result) > limit:
        result = rng.choice(result, limit, replace=False)
    rng.shuffle(result)
    return result


def to_float(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    return images.astype(np.float32) / (255.0 if images.dtype == np.uint8 else 1.0)


def rgb_to_gray(images: np.ndarray) -> np.ndarray:
    images = to_float(images)
    if images.shape[-1] != 3:
        raise ValueError("Ảnh RGB phải có 3 kênh ở chiều cuối")
    weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    return np.tensordot(images, weights, axes=([-1], [0]))


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
            raise RuntimeError("StandardScaler chưa được fit")
        return (np.asarray(features, dtype=np.float32) - self.mean_) / self.scale_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
