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

Tài liệu đầy đủ về kiến trúc, luồng xử lý, format artifact và công dụng từng file nằm tại [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
