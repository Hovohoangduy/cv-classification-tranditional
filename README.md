# CIFAR-10 Classification with Traditional Computer Vision

Project phân loại CIFAR-10 bằng các bộ trích xuất đặc trưng và mô hình học máy được cài đặt từ đầu với NumPy.

## Thành phần đã triển khai

- Tải/đọc CIFAR-10 bản Python chính thức và kiểm tra MD5.
- Chia train/validation phân tầng, không sử dụng test để fit preprocessing.
- HOG: gradient, orientation voting và L2-Hys.
- SIFT mức học thuật: Gaussian/DoG, phát hiện extrema, loại biên, gán hướng và descriptor 128 chiều.
- K-means + Bag of Visual Words.
- Linear multiclass SVM với hinge loss và mini-batch SGD.
- CART và Random Forest với bootstrap/random subspace.
- Accuracy, precision, recall, macro-F1 và confusion matrix.
- Lưu feature cache, model, cấu hình, timing và kết quả JSON.
- CLI inference cho ảnh bên ngoài hoặc ảnh theo index trong CIFAR-10 test.

Pipeline chính không gọi thuật toán HOG/SIFT/SVM/Random Forest từ OpenCV hoặc scikit-learn. NumPy chỉ được dùng cho phép toán mảng và đại số tuyến tính.

## Cài đặt

Yêu cầu Python 3.9 trở lên.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Chạy kiểm thử

```bash
python -m pytest -q
```

## Quick start

Lần đầu chạy, dữ liệu CIFAR-10 được tải về thư mục `data/`.

Smoke test nhanh với tập con có phân tầng:

```bash
python scripts/extract_features.py \
  --feature hog \
  --output outputs/features/hog_smoke.npz \
  --train-limit 1000 \
  --test-limit 200
```

Train smoke test bằng cấu hình dành riêng:

```bash
python scripts/train.py --config configs/hog_svm_smoke.json
```

## Chạy các thí nghiệm đầy đủ

### 1. Trích xuất đặc trưng

```bash
python scripts/extract_features.py --feature pixel --output outputs/features/pixel.npz
python scripts/extract_features.py --feature hog --cell-size 4 --output outputs/features/hog.npz
python scripts/extract_features.py --feature sift_bovw --vocabulary-size 128 \
  --output outputs/features/sift_bovw.npz
```

SIFT thuần NumPy có chi phí cao. Nên chạy `--train-limit 1000 --test-limit 200` trước khi dùng toàn bộ dữ liệu.

### 2. Train và đánh giá

```bash
python scripts/run_experiments.py \
  configs/pixel_svm.json \
  configs/hog_svm.json \
  configs/hog_rf.json \
  configs/sift_bovw_svm.json
```

Hoặc chạy từng cấu hình:

```bash
python scripts/train.py --config configs/hog_svm.json
```

Mỗi lần chạy sinh ra:

- model và scaler trong `outputs/models/`;
- metrics/config/timing trong `outputs/metrics/`;
- confusion matrix trong `outputs/figures/`.

Đánh giá lại một model đã lưu:

```bash
python scripts/evaluate.py \
  --model outputs/models/hog_svm.pkl \
  --features outputs/features/hog.npz \
  --split test
```

## Inference bằng model đã train

```bash
python inference.py --cifar-index 0 1 2 --top-k 3
```

Inference trên một hoặc nhiều ảnh bên ngoài:

```bash
python inference.py \
  --model outputs/models/hog_svm.pkl \
  --image path/to/image-1.png path/to/image-2.jpg \
  --top-k 5 \
  --output outputs/inference/result.json
```

Ảnh đầu vào được chuyển sang RGB và resize về 32x32. Script tự đọc loại đặc trưng,
tham số HOG, scaler và classifier từ model bundle. `score` là decision score của
SVM, không phải xác suất đã hiệu chỉnh.

Prediction artifact, loss curve và biểu đồ theo lớp được sinh khi cấu hình có các trường
`predictions_output`, `loss_output` và `per_class_output`. Hai cấu hình đầy đủ
`pixel_svm.json` và `hog_svm.json` đã bật sẵn các trường này.

Sinh gallery và thống kê bằng chứng từ predictions:

```bash
python scripts/generate_evidence.py \
  --predictions outputs/predictions/hog_svm.npz \
  --gallery-output outputs/figures/hog_svm_predictions.png \
  --margin-output outputs/figures/hog_svm_margin.png \
  --summary-output outputs/evidence/hog_svm.json
```

## Quy trình thí nghiệm đúng

1. Dùng validation để so sánh siêu tham số.
2. Chốt một cấu hình.
3. Chỉ dùng kết quả test của cấu hình đã chốt trong báo cáo.
4. Chạy cấu hình cuối với ít nhất ba seed nếu tài nguyên cho phép.

Các file JSON trong `configs/` là điểm khởi đầu, không phải khẳng định rằng các siêu tham số đó tối ưu.

## Cấu trúc chính

```text
src/data/          # CIFAR-10 loader và stratified split
src/features/      # HOG, SIFT, k-means, BoVW
src/models/        # Linear SVM, CART, Random Forest
scripts/           # extract, train, evaluate, experiments
configs/           # cấu hình thí nghiệm
tests/             # unit tests
report/            # báo cáo LaTeX tiếng Việt
outputs/           # cache, model, metrics và figures
```

## Giới hạn đã biết

- SIFT được viết nhằm minh họa thuật toán, sử dụng một octave trên ảnh upscale; không phải bản thay thế tối ưu hóa cho OpenCV SIFT.
- Random Forest from scratch ưu tiên tính dễ đọc và có thể chậm trên 50.000 mẫu HOG.
- SURF không được đưa vào pipeline chính vì ảnh CIFAR-10 chỉ 32×32 và chi phí cài đặt/kiểm chứng lớn so với giá trị bổ sung. Phần này được giải thích trong báo cáo.

Xem lộ trình và thiết kế thí nghiệm đầy đủ tại [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Sinh bảng và biên dịch báo cáo

```bash
python scripts/generate_report_results.py outputs/metrics/*.json
cd report
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Template dùng XeLaTeX để hỗ trợ tiếng Việt Unicode mà không phụ thuộc gói mã hóa T5.
