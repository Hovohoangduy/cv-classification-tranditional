# CIFAR-10 với Computer Vision truyền thống

Đồ án nhỏ phân loại CIFAR-10 bằng HOG, SIFT–Bag of Visual Words, Linear SVM và Random Forest. Các thuật toán chính được cài đặt từ đầu bằng NumPy; không dùng OpenCV hoặc scikit-learn cho feature extractor/classifier.

## Cài đặt bằng Conda

```bash
conda env create -f environment.yml
conda activate cifar10-traditional-cv
```

Khi `requirements.txt` thay đổi, cập nhật môi trường bằng `python -m pip install -r requirements.txt`.

## Chạy nhanh

Các feature/model/kết quả hiện có đã được giữ trong `outputs/`.

```bash
# Inference model HOG-SVM đã train
python inference.py --cifar-index 0 1 2 --top-k 3

# Đánh giá lại model trên feature cache
python main.py evaluate \
  --model outputs/models/hog_svm.pkl \
  --features outputs/features/hog.npz

# Chạy kiểm thử
python -m pytest -q
```

## Live demo bằng ảnh

Tạo 10 ảnh từ tập test chính thức (mỗi lớp một ảnh) và manifest chứa actual label:

```bash
python main.py demo-data
```

Inference nhiều ảnh, lưu plot và mở cửa sổ trình diễn:

```bash
python inference.py \
  --image demo/images/*.png \
  --plot demo/prediction_demo.png \
  --show
```

`inference.py` tự lấy actual label từ `demo/labels.json`. Plot hiển thị ảnh đầu vào, predicted label, actual label và top-k decision score. Khi inference ảnh riêng không có trong manifest, truyền nhãn thật bằng `--actual-label cat` (hoặc số lớp); nếu không biết nhãn, phần actual label sẽ ghi “không cung cấp”. Có thể bỏ `--show` khi chỉ muốn sinh file PNG.

## Chạy thí nghiệm mới

```bash
# 1. Trích xuất HOG
python main.py extract --feature hog --output outputs/features/hog.npz

# 2. Train SVM; các artifact tự động vào outputs/
python main.py train \
  --features outputs/features/hog.npz \
  --name hog_svm --model svm \
  --epochs 30 --batch-size 256

# 3. Sinh bằng chứng thực nghiệm
python main.py evidence \
  --predictions outputs/predictions/hog_svm.npz \
  --name hog_svm
```

Để smoke test nhanh, thêm `--train-limit 1000 --test-limit 200` khi extract. SIFT thuần NumPy chậm hơn đáng kể, nên luôn smoke test trước.

## Inference ảnh riêng

```bash
python inference.py \
  --model outputs/models/hog_svm.pkl \
  --image path/to/image.jpg \
  --actual-label cat \
  --plot demo/my_prediction.png \
  --top-k 5 \
  --output outputs/inference/result.json
```

Ảnh được đổi sang RGB và resize về 32×32. `score` là decision score của SVM, không phải xác suất.

## Biên dịch báo cáo

```bash
bash report/build.sh
```

PDF được tạo trực tiếp tại `report/cifar10_traditional_cv_report.pdf`. Script tự dọn file phụ trợ LaTeX và không tạo thêm thư mục `output/`.

Tạo lại bốn sơ đồ kiến trúc theo cài đặt CIFAR-10 bằng `python -m report.generate_architectures`. Sơ đồ được xuất dưới dạng PDF vector và PNG tại `report/assets/figures/`.

Tạo lại hình ảnh mẫu của 10 lớp CIFAR-10 bằng `python -m report.generate_dataset_samples`; các ảnh được lấy từ tập huấn luyện chính thức.

Để tạo lại ảnh minh họa từ bốn checkpoint đã lưu trong `outputs/deep_models_200_gn/`, chạy `python -m report.generate_visuals` trước khi biên dịch. Mỗi mô hình chỉ minh họa một ảnh đúng và một ảnh sai, chọn mẫu ở giữa mỗi nhóm sau khi sắp theo cross-entropy trên đủ 10.000 ảnh test. Index, điểm xếp hạng và nguồn checkpoint được lưu trong `report/assets/results/prediction_examples.json`; thao tác này chỉ suy luận, không huấn luyện lại.

Tài liệu đầy đủ về kiến trúc, luồng xử lý, format artifact và công dụng từng file nằm tại [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Thí nghiệm học sâu độc lập

AlexNet, VGG11, ResNet18 và ViT tiny cho ảnh CIFAR-10 được định nghĩa từ đầu trong `deep_models/models.py`; không dùng torchvision models hoặc pretrained weights. Cài thêm PyTorch bằng `python -m pip install -r requirements-deep.txt`.

AlexNet dùng GroupNorm sau các lớp convolution và toàn bộ mô hình được giới hạn gradient norm để ổn định huấn luyện trên MPS. Checkpoint AlexNet cũ sinh trước thay đổi này không tương thích; hãy chạy lại từ đầu và không dùng `--resume` với checkpoint cũ.

```bash
# Kiểm tra nhanh toàn bộ pipeline trên một phần dữ liệu
bash run_deep_models.sh --quick

# Thí nghiệm đầy đủ: 45.000 train, 5.000 validation, 10.000 test, 200 epochs
bash run_deep_models.sh
```

Script nhận thêm các tùy chọn như `--epochs`, `--batch-size`, `--device`, `--models` và `--train-limit`; dùng `--resume` để tiếp tục lần chạy dài bị gián đoạn. Checkpoint và metrics mặc định nằm trong `outputs/deep_models_200_gn/`, tách khỏi checkpoint AlexNet cũ đã mất ổn định; `--quick` dùng riêng `outputs/deep_models_quick/`. Sau khi đủ bốn mô hình hoàn tất 200 epoch, script cập nhật [báo cáo riêng](report/deep_models_report.md), tạo biểu đồ so sánh với các metrics SVM 200 epoch và Random Forest 200 cây, thêm kết quả vào PDF rồi biên dịch lại `report/cifar10_traditional_cv_report.pdf`. Kết quả `--quick` chỉ kiểm tra luồng chạy, không phải phép so sánh cuối với thí nghiệm đầy đủ.

Trên CPU hiện tại, lần chạy 2 epoch cho cả bốn mô hình mất khoảng 24 phút; 200 epoch có thể cần khoảng 40 giờ. Đây là ước lượng, không phải thời gian bảo đảm. `--resume` tiếp tục từ checkpoint cuối mỗi epoch; biểu đồ so sánh chỉ xuất hiện sau khi có đủ số liệu 200 epoch.
