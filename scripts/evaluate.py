#!/usr/bin/env python3
"""Evaluate an already trained model bundle on one cached split."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    with open(args.model, "rb") as stream:
        bundle = pickle.load(stream)
    archive = np.load(args.features, allow_pickle=False)
    features = bundle["scaler"].transform(archive[f"x_{args.split}"])
    predictions = bundle["model"].predict(features)
    metrics = classification_metrics(archive[f"y_{args.split}"], predictions, len(bundle["class_names"]))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
