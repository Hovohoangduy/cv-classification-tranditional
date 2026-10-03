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
    rows = [r"% Traditional baselines: outputs/metrics/*_200.json"]
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
    baseline = next(item for item in traditional if item["feature"] == "hog" and
                    item["model"] == "linear_svm")
    baseline_errors = round((1 - baseline["test"]["accuracy"]) * baseline["samples"]["test"])
    gain_rows = []
    for item in sorted(results, key=lambda item: item["test"]["accuracy"], reverse=True):
        errors = round((1 - item["test"]["accuracy"]) * item["samples"]["test"])
        accuracy_gain = 100 * (item["test"]["accuracy"] - baseline["test"]["accuracy"])
        f1_gain = 100 * (item["test"]["macro_f1"] - baseline["test"]["macro_f1"])
        reduction = 100 * (baseline_errors - errors) / baseline_errors
        gain_rows.append(f"{display_names[item['model']].split(' (')[0]} & {accuracy_gain:.2f} & "
                         f"{f1_gain:.2f} & {errors:,} & {reduction:.2f}" + r"\% \\")
    gain_rows[-1] = gain_rows[-1].rstrip("\\ ")
    (report_dir / "assets" / "results" / "deep_gains.tex").write_text(
        "\n".join(gain_rows) + "\n", encoding="utf-8")
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
    display_type = "kiểm tra nhanh" if run_type == "quick smoke test" else "thí nghiệm theo cấu hình"
    if any(item["samples"] != sample_counts or item["epochs"] != epochs for item in results):
        raise ValueError("Cannot combine experiments with different samples or epochs")
    alexnet_has_groupnorm = any(item["model"] == "alexnet" and
                               "GroupNorm" in item["architecture"] for item in results)
    alexnet_description = ("AlexNet dùng năm convolution với GroupNorm và classifier hai lớp; " if
                           alexnet_has_groupnorm else
                           "AlexNet dùng năm convolution và classifier hai lớp; ")
    report_dir = Path("report")
    result_dir = report_dir / "assets" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = _comparison_200(results, report_dir)
    if comparison_path is not None:
        _deep_diagnostics(results, report_dir)
    md = ["# CIFAR-10: AlexNet, VGG11, ResNet18 và ViT", "",
          "Các mô hình được định nghĩa từ đầu bằng các lớp cơ bản của PyTorch, "
          "khởi tạo trọng số ngẫu nhiên và huấn luyện riêng. Không dùng pretrained/model zoo.", "",
          "Các biến thể cho ảnh 32×32: " + alexnet_description +
          "VGG11-BN giữ tám convolution và năm pooling nhưng giảm số kênh; "
          "ResNet18 giữ tám residual block với stem 3×3 và base width 32; "
          "ViT tiny dùng patch 4×4, embedding 192, sáu block và ba attention head.", "",
          f"Loại chạy: **{display_type}**. Seed: **{results[0]['seed']}**. "
          f"Epoch tối đa: **{epochs}**. "
          f"Mẫu train/validation/test: **{sample_counts['train']}/{sample_counts['validation']}/{sample_counts['test']}**. "
          f"Thiết bị: **{results[0]['device']}**.", "",
          "Tập validation được tách phân tầng từ 50.000 ảnh train chính thức. "
          "Trọng số tốt nhất được chọn bằng accuracy validation; tập test chỉ được đánh giá sau đó. "
          "Chuẩn hóa dùng thống kê của tập train đã chọn. Không tăng cường dữ liệu.", "",
          "| Mô hình | Số tham số | Epoch tốt nhất | Validation accuracy | Test accuracy | Test macro-F1 | Giây train |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    tex_rows = []
    for item in results:
        md.append(f"| {item['model']} | {item['parameters']:,} | {item['best_epoch']} | "
                  f"{item['validation']['accuracy']:.4f} | {item['test']['accuracy']:.4f} | "
                  f"{item['test']['macro_f1']:.4f} | {item['training_seconds']:.1f} |")
        name = {"alexnet": "AlexNet (GroupNorm)", "vgg11": "VGG11-BN",
                "resnet18": "ResNet18", "vit_tiny": "ViT tiny"}[item["model"]]
        tex_rows.append(f"{name} & {item['parameters']:,} & {item['best_epoch']} & "
                        f"{item['validation']['accuracy']:.4f} & "
                        f"{item['training_seconds'] / 3600:.2f}")
    scope_note = ("Đây là kiểm tra nhanh trên một phần dữ liệu; không dùng các số liệu này "
                  "để so sánh hiệu năng cuối cùng với thí nghiệm 10.000 ảnh test." if
                  run_type == "quick smoke test" else
                  "Các kết quả này được đo trên đủ 10.000 ảnh test chính thức với "
                  "đúng cấu hình huấn luyện ghi trên.")
    md.extend(["", scope_note, "",
               "Mã chạy: `bash run_deep_models.sh` (đầy đủ) hoặc "
               "`bash run_deep_models.sh --quick` (kiểm tra luồng). "
               f"JSON và checkpoint của lần chạy này nằm trong `{Path(metric_paths[0]).parent}/`.", ""])
    if epochs < 10 and run_type != "quick smoke test":
        md.extend([f"Lưu ý: mới huấn luyện {epochs} epoch; chưa xác nhận các mô hình đã hội tụ.", ""])
    if comparison_path is not None:
        md.extend(["## So sánh với phương pháp truyền thống", "",
                   "![Accuracy và macro-F1 trên CIFAR-10](assets/figures/deep_vs_traditional_200.png)", "",
                   "SVM và bốn mạng được huấn luyện 200 epoch; Random Forest dùng 200 cây. "
                   "Các mô hình dùng cùng cỡ tập dữ liệu và seed 42, nhưng chi phí tính toán "
                   "và thuật toán tối ưu khác nhau.", ""])
    (report_dir / "deep_models_report.md").write_text("\n".join(md), encoding="utf-8")
    note = ("Đây chỉ là smoke test quy mô nhỏ; các số liệu không được so sánh trực tiếp "
            "với bảng thí nghiệm truyền thống dùng toàn bộ 10.000 ảnh test." if run_type == "quick smoke test" else
            "Các số liệu tương ứng đúng với cấu hình và tập mẫu được ghi ở đây. " +
            (f"Mới huấn luyện {epochs} epoch; chưa xác nhận mô hình đã hội tụ." if epochs < 10 else ""))
    tex = [r"\subsection{Kết quả huấn luyện các mô hình học sâu}",
           r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{Checkpoint và chi phí huấn luyện các mô hình học sâu; accuracy trong khoảng $[0,1]$.}",
           r"\label{tab:deep-training}",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Mô hình & Tham số & Epoch tốt & Val acc. & Train (giờ) \\",
           r"\midrule", (r" \\" + "\n").join(tex_rows) + r" \\", r"\bottomrule",
           r"\end{tabular}", r"\end{table}", note]
    if comparison_path is not None:
        tex.extend([r"\begin{figure}[H]", r"\centering",
                    r"\includegraphics[width=0.95\linewidth]{assets/figures/deep_vs_traditional_200.png}",
                    r"\caption{Accuracy và macro-F1 trên cùng 10.000 ảnh test. SVM và bốn mạng dùng 200 epoch; Random Forest dùng 200 cây.}",
                    r"\label{fig:comparison}",
                    r"\end{figure}",
                    r"Các kết quả có cùng cỡ tập dữ liệu và seed 42, nhưng thuật toán tối ưu và chi phí tính toán khác nhau."])
    if comparison_path is not None:
        (result_dir / "deep_models_section.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        print(f"Updated {report_dir / 'deep_models_report.md'} and integrated comparison artifacts")
    else:
        print(f"Updated {report_dir / 'deep_models_report.md'}; integrated PDF artifacts retained "
              "because comparison requires all four full-data 200-epoch runs")
