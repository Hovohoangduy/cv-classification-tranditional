#!/usr/bin/env python3
"""Extract and cache pixel, HOG, or SIFT-BoVW features."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_cifar10, stratified_split
from src.features import BagOfVisualWords, HOG, SIFT
from src.preprocessing import to_float
from src.utils import limit_stratified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=("pixel", "hog", "sift_bovw"), required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--cell-size", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=9)
    parser.add_argument("--vocabulary-size", type=int, default=128)
    return parser.parse_args()


def pixel_features(images: np.ndarray) -> np.ndarray:
    return to_float(images).reshape(len(images), -1)


def sift_descriptors(images: np.ndarray, extractor: SIFT, split_name: str) -> list[np.ndarray]:
    descriptors: list[np.ndarray] = []
    for index, image in enumerate(images):
        descriptors.append(extractor.detect_and_describe(image)[1])
        if (index + 1) % 500 == 0:
            print(f"SIFT {split_name}: {index + 1}/{len(images)}")
    return descriptors


def main() -> None:
    args = parse_args()
    dataset = load_cifar10(args.data_dir)
    train_idx, validation_idx = stratified_split(dataset.train_labels, seed=args.seed)
    train_idx = limit_stratified(train_idx, dataset.train_labels, args.train_limit, args.seed)
    validation_limit = None if args.train_limit is None else max(10, args.train_limit // 9)
    validation_idx = limit_stratified(validation_idx, dataset.train_labels, validation_limit, args.seed + 1)
    test_indices = np.arange(len(dataset.test_labels))
    test_indices = limit_stratified(test_indices, dataset.test_labels, args.test_limit, args.seed + 2)
    image_sets = (
        dataset.train_images[train_idx],
        dataset.train_images[validation_idx],
        dataset.test_images[test_indices],
    )

    started = time.perf_counter()
    if args.feature == "pixel":
        feature_sets = tuple(pixel_features(images) for images in image_sets)
    elif args.feature == "hog":
        extractor = HOG(cell_size=args.cell_size, num_bins=args.num_bins)
        feature_sets = tuple(extractor.transform(images, verbose=True) for images in image_sets)
    else:
        extractor = SIFT()
        train_descriptors = sift_descriptors(image_sets[0], extractor, "train")
        validation_descriptors = sift_descriptors(image_sets[1], extractor, "validation")
        test_descriptors = sift_descriptors(image_sets[2], extractor, "test")
        vocabulary = BagOfVisualWords(vocabulary_size=args.vocabulary_size)
        vocabulary.fit(train_descriptors)
        feature_sets = (
            vocabulary.transform(train_descriptors),
            vocabulary.transform(validation_descriptors),
            vocabulary.transform(test_descriptors),
        )
    elapsed = time.perf_counter() - started

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        x_train=feature_sets[0],
        y_train=dataset.train_labels[train_idx],
        x_validation=feature_sets[1],
        y_validation=dataset.train_labels[validation_idx],
        x_test=feature_sets[2],
        y_test=dataset.test_labels[test_indices],
        train_indices=train_idx,
        validation_indices=validation_idx,
        test_indices=test_indices,
        class_names=np.asarray(dataset.class_names),
        extraction_seconds=np.asarray(elapsed),
        feature_name=np.asarray(args.feature),
    )
    print(f"Saved {args.feature} features with shape {feature_sets[0].shape} to {destination}")


if __name__ == "__main__":
    main()
