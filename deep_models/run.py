"""Train independently implemented CNN/ViT models on CIFAR-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import MODELS, create_model

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog",
               "frog", "horse", "ship", "truck")


def read_cifar(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the official Python batches, independently of the traditional pipeline."""
    import pickle

    folder = root / "cifar-10-batches-py"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing {folder}. Download CIFAR-10 before running.")

    def batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open("rb") as file:
            data = pickle.load(file, encoding="bytes")
        images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images, np.asarray(data[b"labels"], dtype=np.int64)

    train = [batch(folder / f"data_batch_{i}") for i in range(1, 6)]
    test_images, test_labels = batch(folder / "test_batch")
    return (np.concatenate([part[0] for part in train]),
            np.concatenate([part[1] for part in train]), test_images, test_labels)


def split_indices(labels: np.ndarray, seed: int, validation_fraction: float = 0.1
                  ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = round(len(indices) * validation_fraction)
        train.extend(indices[count:])
        validation.extend(indices[:count])
    return rng.permutation(train), rng.permutation(validation)


def balanced_limit(indices: np.ndarray, labels: np.ndarray, limit: int | None,
                   seed: int) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    if limit < 10:
        raise ValueError("Each sample limit must allow at least one image per class (>=10)")
    rng = np.random.default_rng(seed)
    groups = []
    for label in range(10):
        candidates = indices[labels[indices] == label]
        count = limit // 10 + (1 if label < limit % 10 else 0)
        groups.append(rng.choice(candidates, count, replace=False))
    return rng.permutation(np.concatenate(groups))


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool,
                seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(images)),
                            torch.from_numpy(np.ascontiguousarray(labels)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def normalize(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
              device: torch.device) -> torch.Tensor:
    return (images.to(device).permute(0, 3, 1, 2).float() / 255.0 - mean) / std


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    matrix = np.zeros((10, 10), dtype=np.int64)
    total_loss = 0.0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            inputs = normalize(images, mean, std, device)
            targets = labels.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * len(labels)
            predictions = logits.argmax(1).cpu().numpy()
            np.add.at(matrix, (labels.numpy(), predictions), 1)
            total += len(labels)
    diagonal = np.diag(matrix).astype(float)
    precision = np.divide(diagonal, matrix.sum(0), out=np.zeros(10), where=matrix.sum(0) != 0)
    recall = np.divide(diagonal, matrix.sum(1), out=np.zeros(10), where=matrix.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=precision + recall != 0)
    return {"loss": total_loss / total, "accuracy": float(diagonal.sum() / total),
            "macro_f1": float(f1.mean()), "per_class_f1": f1.tolist(),
            "confusion_matrix": matrix.tolist()}


def run(args: argparse.Namespace) -> list[Path]:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if args.quick:
        args.epochs = 1
        args.train_limit = 1000
        args.validation_limit = 200
        args.test_limit = 500
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    all_train_images, all_train_labels, all_test_images, all_test_labels = read_cifar(Path(args.data_dir))
    train_ids, validation_ids = split_indices(all_train_labels, args.seed)
    train_ids = balanced_limit(train_ids, all_train_labels, args.train_limit, args.seed + 1)
    validation_ids = balanced_limit(validation_ids, all_train_labels, args.validation_limit, args.seed + 2)
    test_ids = balanced_limit(np.arange(len(all_test_labels)), all_test_labels,
                              args.test_limit, args.seed + 3)
    split_digest = hashlib.sha256(train_ids.tobytes() + validation_ids.tobytes() +
                                  test_ids.tobytes()).hexdigest()
    train_images, train_labels = all_train_images[train_ids], all_train_labels[train_ids]
    validation_images, validation_labels = all_train_images[validation_ids], all_train_labels[validation_ids]
    test_images, test_labels = all_test_images[test_ids], all_test_labels[test_ids]
    # Training split alone determines normalization statistics.
    pixel_stats = train_images.astype(np.float32) / 255.0
    channel_mean = pixel_stats.mean((0, 1, 2))
    channel_std = pixel_stats.std((0, 1, 2)).clip(min=1e-6)
    del pixel_stats, all_train_images, all_test_images
    mean = torch.tensor(channel_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(channel_std, device=device).view(1, 3, 1, 1)
    validation_loader = make_loader(validation_images, validation_labels, args.batch_size, False, args.seed)
    test_loader = make_loader(test_images, test_labels, args.batch_size, False, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.models:
        print(f"\nTraining {name}: {len(train_ids)} train / {len(validation_ids)} validation / "
              f"{len(test_ids)} test; {args.epochs} epochs on {device}", flush=True)
        torch.manual_seed(args.seed)
        model = create_model(name).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        loader = make_loader(train_images, train_labels, args.batch_size, True, args.seed)
        best_accuracy = -1.0
        best_epoch = 0
        history = []
        checkpoint_path = output / f"{name}.pt"
        latest_path = output / f"{name}_latest.pt"
        config = {"model": name, "seed": args.seed, "device": str(device),
                  "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                  "weight_decay": args.weight_decay, "split_digest": split_digest,
                  "max_grad_norm": args.max_grad_norm, "model_revision": "alexnet_groupnorm_v2" if
                  name == "alexnet" else "initial_v1"}
        first_epoch = 1
        previous_seconds = 0.0
        if args.resume and latest_path.exists():
            state = torch.load(latest_path, map_location=device, weights_only=True)
            if state["config"] != config:
                raise ValueError(
                    f"Checkpoint {latest_path} is incompatible with the current model or settings; "
                    "start a fresh run without --resume in a new output directory"
                )
            if state["epoch"] > args.epochs:
                raise ValueError(f"Saved epoch {state['epoch']} exceeds requested {args.epochs}")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            loader.generator.set_state(state["loader_rng"].cpu())
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "mps" and state.get("device_rng") is not None:
                torch.mps.set_rng_state(state["device_rng"].cpu())
            elif device.type == "cuda" and state.get("device_rng") is not None:
                torch.cuda.set_rng_state_all([value.cpu() for value in state["device_rng"]])
            history = state["history"]
            best_accuracy = state["best_accuracy"]
            best_epoch = state["best_epoch"]
            previous_seconds = state["training_seconds"]
            first_epoch = state["epoch"] + 1
            print(f"Resuming {name} from epoch {state['epoch']}", flush=True)
        elif args.resume:
            print(f"No latest checkpoint for {name}; starting from epoch 1", flush=True)
        started = time.perf_counter()
        for epoch in range(first_epoch, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_number, (images, labels) in enumerate(loader, 1):
                inputs = normalize(images, mean, std, device)
                targets = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                loss_value = loss.item()
                if not np.isfinite(loss_value) or loss_value > 20:
                    raise RuntimeError(
                        f"Unstable loss {loss_value:.4f} in {name}, epoch {epoch}, batch {batch_number}; "
                        "stop this run and restart with a clean checkpoint"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if not torch.isfinite(gradient_norm).item():
                    raise RuntimeError(f"Non-finite gradient in {name}, epoch {epoch}, batch {batch_number}")
                optimizer.step()
                train_loss += loss_value * len(labels)
            validation = evaluate(model, validation_loader, criterion, mean, std, device)
            history.append({"epoch": epoch, "train_loss": train_loss / len(train_labels),
                            "validation_loss": validation["loss"],
                            "validation_accuracy": validation["accuracy"]})
            print(f"{name} epoch {epoch}/{args.epochs}: train loss {history[-1]['train_loss']:.4f}, "
                  f"validation accuracy {validation['accuracy']:.4f}", flush=True)
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                torch.save({"model": model.state_dict(), "model_name": name,
                            "class_names": CLASS_NAMES, "mean": channel_mean.tolist(),
                            "std": channel_std.tolist(), "seed": args.seed,
                            "epoch": epoch}, checkpoint_path)
                best_epoch = epoch
            latest_state = {"config": config, "epoch": epoch,
                            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "loader_rng": loader.generator.get_state(),
                            "torch_rng": torch.get_rng_state(), "history": history,
                            "device_rng": (torch.mps.get_rng_state() if device.type == "mps" else
                                           torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
                            "best_accuracy": best_accuracy, "best_epoch": best_epoch,
                            "training_seconds": previous_seconds + time.perf_counter() - started}
            temporary_path = latest_path.with_suffix(".tmp")
            torch.save(latest_state, temporary_path)
            temporary_path.replace(latest_path)
        training_seconds = previous_seconds + time.perf_counter() - started
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        validation = evaluate(model, validation_loader, criterion, mean, std, device)
        test = evaluate(model, test_loader, criterion, mean, std, device)
        result = {
            "model": name, "architecture": {"alexnet": "AlexNet CIFAR with GroupNorm",
                "vgg11": "VGG11-BN CIFAR (reduced width)",
                "resnet18": "ResNet18 CIFAR (base width 32)",
                "vit_tiny": "ViT tiny (patch 4, dim 192, depth 6, heads 3)"}[name],
            "parameters": parameters, "seed": args.seed, "device": str(device),
            "epochs": args.epochs, "best_epoch": best_epoch,
            "samples": {"train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids)},
            "split": "stratified 90/10 of official training set; official test set held out",
            "normalization": {"mean": channel_mean.tolist(), "std": channel_std.tolist()},
            "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                          "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                          "max_grad_norm": args.max_grad_norm},
            "train_augmentation": "none", "history": history, "validation": validation,
            "test": test, "training_seconds": training_seconds,
            "checkpoint": str(checkpoint_path),
            "environment": {"python": platform.python_version(), "torch": torch.__version__},
            "run_type": "quick smoke test" if args.quick else "configured experiment",
        }
        metric_path = output / f"{name}_metrics.json"
        metric_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(metric_path)
        print(f"{name}: test accuracy {test['accuracy']:.4f}, macro-F1 {test['macro_f1']:.4f} "
              f"({training_seconds:.1f}s training)", flush=True)
        del model, optimizer
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/deep_models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--quick", action="store_true", help="1 epoch on 1000/200/500 samples")
    parser.add_argument("--resume", action="store_true", help="resume from the latest epoch checkpoint")
    args = parser.parse_args()
    paths = run(args)
    from .report import generate_report
    all_paths = [Path(args.output_dir) / f"{name}_metrics.json" for name in MODELS]
    if all(path.exists() for path in all_paths):
        all_results = [json.loads(path.read_text(encoding="utf-8")) for path in all_paths]
        if (len({item["epochs"] for item in all_results}) == 1 and
                len({item["seed"] for item in all_results}) == 1 and
                len({tuple(item["samples"].values()) for item in all_results}) == 1):
            paths = all_paths
    generate_report(paths)


if __name__ == "__main__":
    main()
