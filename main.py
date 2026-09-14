#!/usr/bin/env python3
"""CLI duy nhất để chạy toàn bộ thí nghiệm CIFAR-10 truyền thống."""

from __future__ import annotations

import argparse
import json

from src.data import export_demo_images
from src.pipeline import evaluate, extract_archive, generate_evidence, generate_report_table, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="Trích xuất và cache đặc trưng")
    extract.add_argument("--feature", choices=("pixel", "hog", "sift", "sift_bovw"), required=True)
    extract.add_argument("--output", help="Mặc định outputs/features/<feature>.npz")
    extract.add_argument("--data-dir", default="data")
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--train-limit", type=int)
    extract.add_argument("--test-limit", type=int)
    extract.add_argument("--cell-size", type=int, default=4)
    extract.add_argument("--block-size", type=int, default=2)
    extract.add_argument("--num-bins", type=int, default=9)
    extract.add_argument("--vocabulary-size", type=int, default=128)

    train = commands.add_parser("train", help="Huấn luyện và lưu toàn bộ kết quả")
    train.add_argument("--features", required=True)
    train.add_argument("--name", required=True, help="Tên experiment và prefix artifact")
    train.add_argument("--model", choices=("svm", "rf"), default="svm")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--regularization", type=float, default=1e-4)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--num-trees", type=int, default=50)
    train.add_argument("--max-depth", type=int, default=15)
    train.add_argument("--min-samples-split", type=int, default=4)
    train.add_argument("--min-samples-leaf", type=int, default=2)
    train.add_argument("--num-thresholds", type=int, default=16)
    train.add_argument("--max-features", default="sqrt")
    train.add_argument("--cell-size", type=int, default=4)
    train.add_argument("--block-size", type=int, default=2)
    train.add_argument("--num-bins", type=int, default=9)
    train.add_argument("--verbose", action="store_true")

    check = commands.add_parser("evaluate", help="Đánh giá model đã lưu")
    check.add_argument("--model", required=True)
    check.add_argument("--features", required=True)
    check.add_argument("--split", choices=("validation", "test"), default="test")

    evidence = commands.add_parser("evidence", help="Sinh gallery, margin và thống kê lỗi")
    evidence.add_argument("--predictions", required=True)
    evidence.add_argument("--name", required=True)
    evidence.add_argument("--data-dir", default="data")

    table = commands.add_parser("report-table", help="Sinh bảng LaTeX từ metrics JSON")
    table.add_argument("metrics", nargs="+")
    table.add_argument("--output", default="report/assets/results/generated_results.tex")

    demo = commands.add_parser("demo-data", help="Xuất ảnh CIFAR-10 test có nhãn cho live demo")
    demo.add_argument("--data-dir", default="data")
    demo.add_argument("--destination", default="demo/images")
    demo.add_argument("--manifest", default="demo/labels.json")
    demo.add_argument("--samples-per-class", type=int, default=1)
    demo.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        output = args.output or f"outputs/features/{args.feature}.npz"
        extract_archive(args.feature, output, args.data_dir, args.seed, args.train_limit, args.test_limit,
                        args.cell_size, args.block_size, args.num_bins, args.vocabulary_size)
    elif args.command == "train":
        if args.model == "svm":
            parameters = {"learning_rate": args.learning_rate, "regularization": args.regularization,
                          "epochs": args.epochs, "batch_size": args.batch_size,
                          "seed": args.seed, "verbose": args.verbose}
        else:
            max_features: str | float | int = args.max_features
            if args.max_features != "sqrt":
                max_features = float(args.max_features) if "." in args.max_features else int(args.max_features)
            parameters = {"num_trees": args.num_trees, "max_depth": args.max_depth,
                          "min_samples_split": args.min_samples_split, "min_samples_leaf": args.min_samples_leaf,
                          "max_features": max_features, "num_thresholds": args.num_thresholds,
                          "seed": args.seed, "verbose": args.verbose}
        feature_parameters = {"cell_size": args.cell_size, "block_size": args.block_size, "num_bins": args.num_bins}
        train_experiment(args.features, args.name, args.model, parameters, feature_parameters)
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.model, args.features, args.split), indent=2, ensure_ascii=False))
    elif args.command == "evidence":
        print(json.dumps(generate_evidence(args.predictions, args.name, args.data_dir), indent=2, ensure_ascii=False))
    elif args.command == "report-table":
        generate_report_table(args.metrics, args.output)
        print(f"Đã sinh bảng {args.output}")
    else:
        manifest = export_demo_images(args.data_dir, args.destination, args.manifest,
                                      args.samples_per_class, args.seed)
        print(f"Đã xuất ảnh demo vào {args.destination} và nhãn vào {manifest}")


if __name__ == "__main__":
    main()
