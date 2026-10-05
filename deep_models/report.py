"""Generate neural-network summaries and the integrated CIFAR-10 comparison."""

from __future__ import annotations

import json
from pathlib import Path


def _comparison_200(results: list[dict], report_dir: Path) -> Path | None:
    """Plot measured 200-epoch networks against existing traditional 200 runs."""
    names = ("alexnet", "vgg11", "resnet18", "vit_tiny")
    if ({item["model"] for item in results} != set(names) or
            any(item["epochs"] != 200 or item["samples"] !=
                {"train": 45000, "validation": 5000, "test": 10000} for item in results)):
        return None
    baselines = (
        ("Pixel-SVM", "pixel_svm_200"), ("Pixel-RF", "pixel_rf_200"),
        ("HOG-SVM", "hog_svm_200"), ("HOG-RF", "hog_rf_200"),
        ("SIFT-SVM", "sift_svm_200"), ("SIFT-RF", "sift_rf_200"),
    )
    paths = [Path("outputs/metrics") / f"{key}.json" for _, key in baselines]
    if any(not path.exists() for path in paths):
        print("Traditional 200-run metrics missing; comparison chart skipped")
        return None
    traditional = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for item in traditional:
        parameters = item["model_parameters"]
        if (item["samples"] != results[0]["samples"] or item["seed"] != results[0]["seed"] or
                (parameters.get("epochs") if item["model"] == "linear_svm" else
                 parameters.get("num_trees")) != 200):
            print("Traditional metrics are incompatible; comparison chart skipped")
            return None
    # Keep the common table and chart tied to exactly the same measured runs.
    rows = [r"% Traditional baselines"]
    for (label, _), item in zip(baselines, traditional):
        rows.append(f"Truyền thống & {label} & {item['samples']['test']} & "
                    f"{item['test']['accuracy']:.4f} & {item['test']['macro_f1']:.4f}" + r" \\")
    rows.append(r"\midrule")
    display_names = {"alexnet": "AlexNet (GroupNorm)", "vgg11": "VGG11-BN",
                     "resnet18": "ResNet18", "vit_tiny": "ViT tiny"}
    for name in names:
        item = next(item for item in results if item["model"] == name)
        rows.append(f"Học sâu & {display_names[name]} & {item['samples']['test']} & "
                    f"{item['test']['accuracy']:.4f} & {item['test']['macro_f1']:.4f}" + r" \\")
    rows[-1] = rows[-1].rstrip("\\ ")
    (report_dir / "assets" / "results" / "comparison_results.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8")
    import os
    os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = traditional + [next(item for item in results if item["model"] == name)
                             for name in names]
    labels = [label for label, _ in baselines] + ["AlexNet", "VGG11", "ResNet18", "ViT tiny"]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(positions - 0.2, [item["test"]["accuracy"] for item in ordered], 0.4,
              label="Accuracy", color="#1769aa")
    axis.barh(positions + 0.2, [item["test"]["macro_f1"] for item in ordered], 0.4,
              label="Macro-F1", color="#e3942b")
    axis.set(yticks=positions, yticklabels=labels, xlim=(0, 1),
             xlabel="Điểm trên 10.000 ảnh test CIFAR-10")
    axis.invert_yaxis()
    axis.axhline(5.5, color="0.5", linewidth=0.8)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()
    path = report_dir / "assets" / "figures" / "deep_vs_traditional_200.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _deep_diagnostics(results: list[dict], report_dir: Path) -> None:
    """Render learning curves and the best network's recorded test matrix."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = report_dir / "assets" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for axis, item in zip(axes.flat, results):
        history = item["history"]
        axis.plot([row["epoch"] for row in history],
                  [row["train_loss"] for row in history], label="Train")
        axis.plot([row["epoch"] for row in history],
                  [row["validation_loss"] for row in history], label="Validation")
        axis.axvline(item["best_epoch"], color="0.4", linestyle="--", linewidth=0.8)
        axis.set(title=item["model"], xlabel="Epoch", ylabel="Cross-entropy")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "deep_learning_curves.png", dpi=180)
    plt.close(figure)

    best = max(results, key=lambda item: item["test"]["accuracy"])
    matrix = np.asarray(best["test"]["confusion_matrix"])
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    figure, axis = plt.subplots(figsize=(7, 6))
    plotted = axis.imshow(matrix, cmap="Blues")
    for i in range(10):
        for j in range(10):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axis.set(xticks=range(10), yticks=range(10), xticklabels=labels, yticklabels=labels,
             xlabel="Predicted label", ylabel="True label", title=best["model"])
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.colorbar(plotted, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(figure_dir / "deep_best_confusion.png", dpi=180)
    plt.close(figure)


def generate_report(metric_paths: list[Path]) -> None:
    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in metric_paths]
    if not results:
        raise ValueError("No model results to report")
    sample_counts = results[0]["samples"]
    epochs = results[0]["epochs"]
    run_type = results[0]["run_type"]
    display_type = "kiểm tra nhanh" if run_type == "quick smoke test" else "thí nghiệm"
    if any(item["samples"] != sample_counts or item["epochs"] != epochs for item in results):
        raise ValueError("Cannot combine experiments with different samples or epochs")
    alexnet_has_groupnorm = any(item["model"] == "alexnet" and
                               "GroupNorm" in item["architecture"] for item in results)
    alexnet_description = ("AlexNet dùng năm lớp tích chập với GroupNorm và bộ phân lớp hai lớp; " if
                           alexnet_has_groupnorm else
                           "AlexNet dùng năm lớp tích chập và bộ phân lớp hai lớp; ")
    report_dir = Path("report")
    result_dir = report_dir / "assets" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = _comparison_200(results, report_dir)
    if comparison_path is not None:
        _deep_diagnostics(results, report_dir)
    md = ["# CIFAR-10: AlexNet, VGG11, ResNet18 và ViT", "",
          "Bốn mạng được cài đặt bằng các lớp cơ bản của PyTorch và huấn luyện "
          "từ trọng số ngẫu nhiên, không dùng trọng số huấn luyện sẵn.", "",
          "Các kiến trúc được điều chỉnh cho ảnh 32×32: " + alexnet_description +
          "VGG11-BN dùng tám lớp tích chập với số kênh giảm; "
          "ResNet18 dùng tám block phần dư, lớp tích chập đầu 3×3 và 32 kênh ban đầu; "
          "ViT tiny dùng patch 4×4, vector 192 chiều, sáu block và ba attention head.", "",
          f"Chế độ: **{display_type}**. Seed: **{results[0]['seed']}**. "
          f"Epoch tối đa: **{epochs}**. "
          f"Số ảnh huấn luyện/kiểm định/kiểm tra: **{sample_counts['train']}/{sample_counts['validation']}/{sample_counts['test']}**. "
          f"Thiết bị: **{results[0]['device']}**.", "",
          "Tập kiểm định được tách phân tầng từ 50.000 ảnh huấn luyện chính thức. "
          "Bộ trọng số tốt nhất được chọn theo accuracy kiểm định, sau đó mới đánh giá trên tập kiểm tra. "
          "Chuẩn hóa dùng thống kê của tập huấn luyện; không dùng tăng cường dữ liệu.", "",
          "| Mô hình | Số tham số | Epoch chọn | Accuracy kiểm định | Accuracy kiểm tra | Macro-F1 kiểm tra | Huấn luyện (giờ) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    tex_rows = []
    for item in results:
        name = {"alexnet": "AlexNet (GroupNorm)", "vgg11": "VGG11-BN",
                "resnet18": "ResNet18", "vit_tiny": "ViT tiny"}[item["model"]]
        if item["model"] == "alexnet" and not alexnet_has_groupnorm:
            name = "AlexNet"
        md.append(f"| {name} | {item['parameters']:,} | {item['best_epoch']} | "
                  f"{item['validation']['accuracy']:.4f} | {item['test']['accuracy']:.4f} | "
                  f"{item['test']['macro_f1']:.4f} | {item['training_seconds'] / 3600:.2f} |")
        tex_rows.append(f"{name} & {item['parameters']:,} & {item['best_epoch']} & "
                        f"{item['validation']['accuracy']:.4f} & "
                        f"{item['training_seconds'] / 3600:.2f}")
    scope_note = ("Kết quả kiểm tra nhanh chỉ dùng để kiểm tra chương trình, "
                  "không đại diện cho thí nghiệm trên toàn bộ dữ liệu." if
                  run_type == "quick smoke test" else
                  "Kết quả được đánh giá trên "
                  f"{sample_counts['test']:,}".replace(",", ".") + " ảnh kiểm tra; "
                  "accuracy và macro-F1 được trình bày trên thang [0,1].")
    md.extend(["", scope_note, ""])
    if epochs < 10 and run_type != "quick smoke test":
        md.extend([f"Lưu ý: mới huấn luyện {epochs} epoch; chưa xác nhận các mô hình đã hội tụ.", ""])
    if comparison_path is not None:
        md.extend(["## So sánh với phương pháp truyền thống", "",
                   "![Accuracy và macro-F1 trên CIFAR-10](assets/figures/deep_vs_traditional_200.png)", "",
                   "Biểu đồ đối chiếu các mạng học sâu với sáu cấu hình truyền thống. "
                   "SVM và các mạng chạy 200 epoch; Random Forest dùng 200 cây. "
                   "Các thí nghiệm dùng cùng số ảnh và seed 42, nhưng khác cách tối ưu và chi phí tính toán.", ""])
    (report_dir / "deep_models_report.md").write_text("\n".join(md), encoding="utf-8")
    note = ("Kết quả kiểm tra nhanh không đại diện cho thí nghiệm trên toàn bộ dữ liệu."
            if run_type == "quick smoke test" else
            (f"Mới huấn luyện {epochs} epoch; chưa xác nhận mô hình đã hội tụ." if epochs < 10 else ""))
    tex = [r"\subsection{Kết quả huấn luyện các mô hình học sâu}",
           r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{Epoch được chọn theo accuracy kiểm định và tổng thời gian huấn luyện 200 epoch.}",
           r"\label{tab:deep-training}",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Mô hình & Tham số & Epoch chọn & Acc. kiểm định & Thời gian (giờ) \\",
           r"\midrule", (r" \\" + "\n").join(tex_rows) + r" \\", r"\bottomrule",
           r"\end{tabular}", r"\end{table}", note]
    if comparison_path is not None:
        fastest = min(results, key=lambda item: item["training_seconds"])
        slowest = max(results, key=lambda item: item["training_seconds"])
        best = max(results, key=lambda item: item["test"]["accuracy"])
        display_names = {"alexnet": "AlexNet", "vgg11": "VGG11-BN",
                         "resnet18": "ResNet18", "vit_tiny": "ViT tiny"}
        hours = lambda item: f"{item['training_seconds'] / 3600:.2f}".replace(".", ",")
        tex.append(
            f"{display_names[fastest['model']]} huấn luyện nhanh nhất ({hours(fastest)} giờ), "
            f"còn {display_names[slowest['model']]} mất nhiều thời gian nhất ({hours(slowest)} giờ). "
            f"{display_names[best['model']]} đạt accuracy cao nhất với thời gian {hours(best)} giờ.")
    if comparison_path is not None:
        (result_dir / "deep_models_section.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        print(f"Updated {report_dir / 'deep_models_report.md'} and integrated comparison artifacts")
    else:
        print(f"Updated {report_dir / 'deep_models_report.md'}; integrated PDF artifacts retained "
              "because comparison requires all four full-data 200-epoch runs")
