# Kiến trúc và luồng xử lý

Tài liệu này mô tả phiên bản đã tinh gọn của đồ án phân loại CIFAR-10 bằng computer vision và machine learning truyền thống. Mục tiêu thiết kế là ít file, dễ lần theo luồng chạy, thuật toán đủ rõ để phục vụ môn học, và không làm mất dữ liệu/kết quả đã sinh.

## 1. Phạm vi đồ án

Pipeline hỗ trợ ba biểu diễn ảnh:

- `pixel`: baseline dùng trực tiếp 3.072 giá trị RGB đã chuẩn hóa;
- `hog`: histogram hướng gradient với block normalization L2-Hys;
- `sift_bovw`: SIFT mức học thuật, gom descriptor bằng K-means, rồi mã hóa Bag of Visual Words.

Hai classifier được cài đặt từ đầu:

- Linear multiclass SVM, hinge loss, mini-batch SGD;
- Random Forest gồm nhiều cây CART, bootstrap và chọn ngẫu nhiên feature khi split.

NumPy đảm nhiệm thao tác mảng/đại số tuyến tính. Matplotlib chỉ vẽ bằng chứng; Pillow chỉ đọc và resize ảnh inference; pytest chỉ dùng để kiểm thử.

## 2. Cấu trúc thư mục

```text
.
├── main.py                 # CLI thí nghiệm duy nhất
├── inference.py            # CLI suy luận ảnh/CIFAR index
├── src/
│   ├── __init__.py         # Public API của package
│   ├── data.py             # CIFAR-10, split và preprocessing
│   ├── features.py         # HOG, SIFT, K-means, BoVW
│   ├── models.py           # Linear SVM, CART, Random Forest
│   └── pipeline.py         # Điều phối, metrics, artifact, biểu đồ
├── tests/
│   └── test_core.py        # Unit/integration tests cốt lõi
├── demo/
│   ├── images/             # Ảnh test CIFAR-10 dùng cho live demo
│   ├── labels.json         # Actual label và test index từng ảnh
│   └── prediction_demo.png # Plot kết quả mẫu
├── docs/
│   └── ARCHITECTURE.md     # Tài liệu hiện tại
├── report/                 # Báo cáo tự chứa: LaTeX, PDF, hình và kết quả
│   ├── build.sh            # Biên dịch PDF ngay trong report/
│   ├── main.tex
│   ├── references.bib
│   ├── cifar10_traditional_cv_report.pdf
│   └── assets/
│       ├── figures/        # 5 hình được nhúng vào PDF
│       └── results/        # Bảng, metrics, evidence, predictions
├── data/                   # CIFAR-10 gốc, được giữ nguyên
├── outputs/                # Cache/model/metrics/figures, giữ nguyên
├── environment.yml         # Khai báo Conda environment
├── requirements.txt        # Python dependencies tối thiểu
└── README.md               # Hướng dẫn chạy nhanh
```

So với cấu trúc cũ, các package con quá nhỏ, bảy script và nhiều file cấu hình JSON đã được hợp nhất. Tham số thí nghiệm hiện được truyền trực tiếp qua `main.py`, giúp một đồ án nhỏ không phải đồng bộ nhiều entry point.

## 3. Luồng dữ liệu tổng quát

```text
data/CIFAR-10
      │
      ▼
load + stratified split
      │
      ▼
pixel | HOG | SIFT → K-means → BoVW
      │
      ├──► outputs/features/*.npz
      │
      ▼
StandardScaler (fit chỉ trên train)
      │
      ▼
Linear SVM | Random Forest
      │
      ├──► outputs/models/*.pkl
      ├──► outputs/metrics/*.json
      ├──► outputs/predictions/*.npz
      └──► outputs/figures/*.png
```

Ba split có vai trò riêng:

1. `train`: fit vocabulary, scaler và classifier;
2. `validation`: chọn đặc trưng/siêu tham số;
3. `test`: chỉ đo cấu hình đã chốt để báo cáo.

Không fit scaler trên validation/test. `test_indices` được lưu cùng predictions để truy ngược đúng ảnh gốc khi tạo gallery bằng chứng.

## 4. Công dụng từng file code

### `main.py`

Entry point duy nhất cho thí nghiệm, gồm `extract`, `train`, `evaluate`, `evidence`, `report-table` và `demo-data`. File này chỉ parse tham số và gọi hàm trong `src.pipeline`/`src.data`; thuật toán không nằm trong CLI. Lệnh `demo-data` tải CIFAR-10 nếu cần rồi xuất ảnh test có nhãn thật.

### `inference.py`

Entry point suy luận nhận một hay nhiều ảnh qua `--image`, hoặc index test CIFAR-10 qua `--cifar-index`. Nó in top-k decision score, có thể xuất JSON, lưu plot bằng `--plot` và mở plot bằng `--show` cho live demo. Với ảnh trong `demo/images/`, actual label được tra tự động từ `demo/labels.json`; ảnh riêng có thể nhận `--actual-label`. Ảnh được đổi RGB và resize 32×32. Loại feature, tham số HOG, scaler và classifier được đọc từ model bundle để tránh lệch train/inference.

### `src/data.py`

Chịu trách nhiệm toàn bộ dữ liệu và preprocessing:

- tải CIFAR-10 chính thức, kiểm tra MD5 và giải nén an toàn;
- đọc Python batch và class names;
- xuất ảnh test cùng manifest actual label cho live demo;
- tạo train/validation split phân tầng;
- lấy tập con phân tầng cho smoke test;
- chuyển `uint8` sang float, RGB sang grayscale;
- `StandardScaler` from scratch.

Scaler lưu `mean_` và `scale_`; cột hằng có scale bằng 1 để tránh chia 0.

### `src/features.py`

Gộp toàn bộ thuật toán đặc trưng:

- `HOG`: gradient, magnitude/orientation, nội suy phiếu giữa hai bin, cell histogram và chuẩn hóa block L2-Hys;
- `SIFT`: Gaussian/DoG, extrema, lọc edge response, orientation histogram và descriptor 128 chiều;
- `KMeans`: lặp assign/update đến khi hội tụ;
- `BagOfVisualWords`: học visual vocabulary từ descriptor train và tạo histogram visual word chuẩn hóa L2.

SIFT là phiên bản giáo dục một octave, ưu tiên diễn giải hơn tốc độ như OpenCV.

### `src/models.py`

Gộp ba cấu trúc mô hình: `LinearSVM` dùng multiclass hinge loss, L2 và SGD; `DecisionTreeClassifier` split theo Gini impurity; `RandomForestClassifier` dùng bootstrap, random feature subspace và majority vote. `Node` là cấu trúc nội bộ của cây. Seed được truyền xuyên suốt để tái lập. File cũng chứa compatibility unpickler để các model cũ vẫn chạy sau khi package được đổi thành `src/`.

### `src/pipeline.py`

Lớp điều phối chứa:

- ghi JSON và điều phối model bundle;
- confusion matrix, accuracy, precision, recall và macro-F1;
- extract/train/evaluate;
- lưu predictions, vẽ confusion/loss/per-class;
- gallery, decision-margin distribution và top confusion pairs;
- sinh bảng kết quả LaTeX.

Gom orchestration ở đây giữ `main.py` ngắn và cho phép test hàm mà không chạy process con.

### `src/__init__.py`

Khai báo public API: loader, preprocessing, feature extractor và classifier chính.

### `tests/test_core.py`

Một file test duy nhất bao phủ split/scaler, HOG/SIFT/BoVW, SVM/CART/Random Forest, metrics, resize ảnh, top-k inference và model bundle. Test dùng dữ liệu tổng hợp nhỏ nên chạy nhanh, không phụ thuộc tải CIFAR-10.

## 5. Môi trường và dependency

`environment.yml` tạo Conda environment `cifar10-traditional-cv`, dùng Python 3.11, cài pip rồi đọc `requirements.txt`. Vì chỉ có một danh sách Python package nên không có hai file dependency lệch nhau.

`requirements.txt` chỉ gồm:

- `numpy`: thuật toán số;
- `matplotlib`: biểu đồ;
- `Pillow`: đọc/resize ảnh inference;
- `pytest`: kiểm thử.

Không cần `.venv`, OpenCV, scikit-learn, pandas hoặc framework deep learning.

## 6. Format và vòng đời artifact

### `outputs/features/<feature>.npz`

Chứa các ma trận/nhãn train-validation-test, index gốc, class names, thời gian extract và tên feature. Đây là cache lớn để train lại mà không extract lại.

### `outputs/models/<experiment>.pkl`

Bundle gồm model, scaler, class names, tên feature và feature parameters. Không unpickle file không đáng tin. Bộ đọc tương thích ánh xạ class của các cấu trúc cũ sang package `src`, vì vậy model hiện có vẫn dùng được.

### `outputs/metrics/<experiment>.json`

Chứa tham số, seed, kích thước dữ liệu, thời gian, metrics validation/test và đường dẫn artifact. Đây là nguồn định lượng chính cho báo cáo.

### `outputs/predictions/<experiment>.npz`

Chứa true/predicted label, decision scores, class names và test indices. Evidence có thể sinh lại mà không inference toàn test set.

### `outputs/figures/` và `outputs/evidence/`

Chứa confusion matrix, loss curve, per-class metrics, prediction gallery, margin distribution và JSON phân tích lỗi.

`data/` và `outputs/` là dữ liệu có giá trị, không thuộc phạm vi dọn tự động. `.gitignore` tiếp tục loại chúng khỏi Git để repository không phình lớn.

### `report/assets/`

Chứa bản sao độc lập của đúng các artifact làm căn cứ cho PDF:

- `figures/`: loss curve, per-class metrics, confusion matrix, margin distribution và prediction gallery;
- `results/`: bảng LaTeX, bốn metrics JSON, evidence JSON và predictions NPZ.

Nhờ đó có thể biên dịch `report/main.tex` mà không phụ thuộc đường dẫn sang `outputs/`. Các bản gốc trong `outputs/` vẫn được giữ theo yêu cầu bảo toàn kết quả.

Biên dịch bằng `bash report/build.sh`. Script chạy XeLaTeX/BibTeX trực tiếp trong `report/`, tạo `report/cifar10_traditional_cv_report.pdf` và tự xóa các file build phụ trợ. Project không dùng thư mục `output/` riêng.

## 7. Các lệnh chuẩn

```bash
# Tạo môi trường
conda env create -f environment.yml
conda activate cifar10-traditional-cv

# Smoke test HOG-SVM
python main.py extract --feature hog \
  --output outputs/features/hog_smoke.npz \
  --train-limit 1000 --test-limit 200
python main.py train --features outputs/features/hog_smoke.npz \
  --name hog_svm_smoke --model svm --epochs 15 --batch-size 128

# Random Forest
python main.py train --features outputs/features/hog_smoke.npz \
  --name hog_rf_smoke --model rf --num-trees 5 --max-depth 8

# SIFT-BoVW smoke test
python main.py extract --feature sift_bovw \
  --output outputs/features/sift_bovw_smoke.npz \
  --vocabulary-size 32 --train-limit 500 --test-limit 100

# Inference và lưu JSON
python inference.py --model outputs/models/hog_svm.pkl \
  --cifar-index 0 1 2 --top-k 3 \
  --output outputs/inference/examples.json

# Chuẩn bị và chạy live demo bằng ảnh
python main.py demo-data
python inference.py --image demo/images/*.png \
  --plot demo/prediction_demo.png --show
```

## 8. Nguyên tắc mở rộng

Để thêm feature: triển khai trong `src/features.py`, thêm nhánh ở `extract_archive`, rồi thêm vào `inference.extract_features` nếu cần ảnh rời. Để thêm classifier: triển khai trong `src/models.py`, thêm lựa chọn CLI và khởi tạo trong `train_experiment`.

Không tạo script/config riêng cho mỗi thí nghiệm nhỏ. Chỉ tách file khi module thực sự khó đọc hoặc có trách nhiệm độc lập rõ ràng.

## 9. Giới hạn đã biết

- CIFAR-10 chỉ 32×32 nên SIFT có ít keypoint và phải upscale; BoVW không chắc vượt HOG.
- SIFT/K-means/Random Forest thuần NumPy ưu tiên minh bạch, không tối ưu cho toàn bộ 50.000 ảnh.
- SVM score chưa được hiệu chỉnh thành xác suất.
- Pickle phụ thuộc Python/NumPy; JSON và NPZ phù hợp hơn để lưu trữ dài hạn.
- SURF không nằm trong pipeline chính vì phức tạp và ít giá trị bổ sung trên ảnh rất nhỏ; phạm vi vẫn đáp ứng feature truyền thống bằng HOG và SIFT.
