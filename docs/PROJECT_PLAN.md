# Kế hoạch thực hiện project phân loại ảnh CIFAR-10 bằng phương pháp truyền thống

## 1. Mục tiêu và phạm vi

### 1.1. Mục tiêu chính

Xây dựng một pipeline phân loại ảnh hoàn chỉnh trên bộ dữ liệu **CIFAR-10**, gồm:

1. Đọc và tiền xử lý dữ liệu.
2. Trích xuất đặc trưng ảnh bằng phương pháp truyền thống.
3. Huấn luyện mô hình machine learning truyền thống.
4. Đánh giá, so sánh và phân tích kết quả.
5. Viết báo cáo bằng **LaTeX, tiếng Việt, không quá 10 trang**.

### 1.2. Ràng buộc triển khai

- Ưu tiên **code from scratch** để thể hiện hiểu biết về thuật toán.
- Chỉ dùng thư viện cho các thao tác nền tảng, không gọi trực tiếp thuật toán hoàn chỉnh từ OpenCV hoặc scikit-learn trong pipeline chính.
- Thư viện dự kiến được phép dùng:
  - `numpy`: đại số tuyến tính và thao tác mảng;
  - `matplotlib`: trực quan hóa kết quả;
  - thư viện chuẩn Python (`pickle`, `tarfile`, `urllib`, `pathlib`, `argparse`, `json`, ...): tải dữ liệu, I/O và cấu hình.
- `scikit-learn` chỉ được dùng trong test kiểm chứng nhỏ (nếu cần), không dùng để tạo kết quả chính trong báo cáo.
- Thiết lập seed và lưu toàn bộ cấu hình để thí nghiệm có thể tái lập.

### 1.3. Phạm vi thuật toán

**Pipeline bắt buộc (MVP):**

- Đặc trưng: HOG tự cài đặt.
- Bộ phân loại: Linear SVM one-vs-rest tự cài đặt.
- Baseline: đặc trưng pixel phẳng + SVM hoặc k-NN đơn giản.

**Pipeline mở rộng ưu tiên:**

- SIFT tự cài đặt ở mức học thuật + Bag of Visual Words (BoVW).
- Random Forest tự cài đặt.
- Kernel SVM với RBF trên tập con dữ liệu nếu tài nguyên cho phép.

**SURF:** chỉ xem là nội dung tùy chọn để thảo luận hoặc cài đặt mở rộng. CIFAR-10 có kích thước ảnh rất nhỏ (32×32), nên số keypoint ổn định thường ít; đồng thời SURF phức tạp và không tạo lợi thế rõ ràng so với HOG/SIFT trong giới hạn thời gian của project.

## 2. Câu hỏi nghiên cứu

Project tập trung trả lời các câu hỏi sau:

1. Đặc trưng HOG cải thiện kết quả thế nào so với vector pixel thô?
2. Linear SVM và Random Forest khác nhau thế nào về accuracy, thời gian huấn luyện và bộ nhớ?
3. Với ảnh nhỏ như CIFAR-10, SIFT + BoVW có hiệu quả hơn HOG không?
4. Những lớp nào dễ nhầm lẫn nhất và nguyên nhân có thể đến từ đâu?
5. Các siêu tham số quan trọng (`cell_size`, số bin HOG, `C`, số cây, độ sâu cây, kích thước vocabulary) ảnh hưởng thế nào đến kết quả?

## 3. Dữ liệu và quy trình đánh giá

### 3.1. Bộ dữ liệu

- CIFAR-10 gồm 60.000 ảnh RGB kích thước 32×32, thuộc 10 lớp.
- Sử dụng đúng tập train chính thức (50.000 ảnh) và test chính thức (10.000 ảnh).
- Tách 10% tập train làm validation bằng phương pháp phân tầng theo lớp.
- Không dùng tập test để chọn siêu tham số.

### 3.2. Tiền xử lý

Các bước dự kiến:

1. Chuyển dữ liệu sang `float32` và chuẩn hóa về `[0, 1]`.
2. Với HOG/SIFT: chuyển RGB sang grayscale theo công thức tự cài đặt.
3. Chuẩn hóa từng chiều đặc trưng bằng mean/std tính **chỉ trên tập train**.
4. Lưu cache đặc trưng `.npz` để tránh tính lại.

Không augmentation trong thí nghiệm chính để giữ đúng trọng tâm phương pháp truyền thống. Có thể thêm horizontal flip như một ablation tùy chọn.

### 3.3. Chỉ số đánh giá

- Accuracy trên test là chỉ số chính.
- Macro precision, macro recall và macro F1.
- Confusion matrix 10×10.
- Thời gian trích xuất đặc trưng, thời gian train và thời gian dự đoán.
- Kích thước vector đặc trưng và dung lượng mô hình.
- Trung bình và độ lệch chuẩn của ít nhất 3 lần chạy với các seed khác nhau cho cấu hình cuối, nếu thời gian cho phép.

## 4. Thiết kế thuật toán from scratch

### 4.1. HOG

Các bước tự cài đặt:

1. Tính gradient theo trục `x`, `y` bằng bộ lọc sai phân.
2. Tính magnitude và orientation không hướng trong khoảng `[0, 180°)`.
3. Chia ảnh thành cell; tạo histogram hướng cho từng cell.
4. Nội suy phiếu bầu giữa hai orientation bin lân cận.
5. Ghép các cell thành block và chuẩn hóa L2-Hys.
6. Trải phẳng các block thành vector đặc trưng.

Siêu tham số khảo sát:

- `cell_size`: 4×4 và 8×8;
- `block_size`: 2×2 cells;
- `num_bins`: 9;
- grayscale so với HOG trên từng kênh màu (tùy chọn).

Kiểm thử:

- Ảnh hằng số phải tạo gradient gần 0.
- Ảnh có cạnh ngang/dọc nhân tạo phải cho orientation bin đúng.
- Kích thước vector đầu ra phải khớp công thức lý thuyết.
- Không xuất hiện `NaN` khi block có norm bằng 0.

### 4.2. SIFT + Bag of Visual Words (mở rộng)

Phần SIFT tự cài đặt dự kiến gồm:

1. Xây dựng Gaussian pyramid và Difference of Gaussians.
2. Tìm extrema theo không gian và tỉ lệ.
3. Loại keypoint có độ tương phản thấp hoặc nằm trên biên mạnh.
4. Gán hướng chính cho keypoint.
5. Tạo descriptor 4×4×8 = 128 chiều.
6. Chuẩn hóa và cắt ngưỡng descriptor.

Do mỗi ảnh CIFAR-10 rất nhỏ, ảnh có thể được upscale lên 64×64 trước khi tìm keypoint; đây phải được ghi rõ như một lựa chọn thực nghiệm.

Để tạo vector có độ dài cố định:

1. Lấy mẫu descriptor từ tập train.
2. Tự cài đặt k-means để tạo visual vocabulary.
3. Gán mỗi descriptor vào visual word gần nhất.
4. Biểu diễn mỗi ảnh bằng histogram BoVW và chuẩn hóa L1/L2.

Kích thước vocabulary khảo sát: 64, 128 hoặc 256 tùy tài nguyên.

### 4.3. Linear SVM đa lớp

- Tự cài đặt binary linear SVM với hinge loss và L2 regularization.
- Tối ưu bằng mini-batch SGD hoặc Pegasos.
- Mở rộng 10 lớp theo chiến lược one-vs-rest.
- Dự đoán lớp có decision score lớn nhất.

Siêu tham số khảo sát:

- `C` hoặc hệ số regularization `lambda`;
- learning rate và lịch giảm learning rate;
- batch size;
- số epoch.

Cần lưu loss theo epoch và kiểm tra loss giảm trên một tập con nhỏ trước khi train toàn bộ dữ liệu.

### 4.4. Random Forest

Tự cài đặt:

1. CART classification tree dùng Gini impurity.
2. Bootstrap sampling cho từng cây.
3. Chọn ngẫu nhiên một tập con feature tại mỗi node.
4. Dừng theo `max_depth`, `min_samples_split` hoặc node thuần.
5. Tổng hợp dự đoán bằng majority voting.

Để tránh chi phí quá lớn với HOG nhiều chiều, mỗi split chỉ khảo sát một số feature và một số threshold ứng viên ngẫu nhiên.

Siêu tham số khảo sát:

- số cây: 20, 50, 100;
- `max_depth`: 10, 20, không giới hạn có kiểm soát;
- `max_features`: `sqrt(d)` hoặc một tỉ lệ của `d`;
- `min_samples_leaf`.

### 4.5. Kernel SVM (tùy chọn)

- Chỉ triển khai sau khi pipeline chính ổn định.
- Có thể dùng SMO đơn giản và RBF kernel.
- Do ma trận kernel có độ phức tạp bộ nhớ theo bình phương số mẫu, chỉ chạy trên tập con có phân tầng và ghi rõ giới hạn này trong báo cáo.

## 5. Cấu trúc repository dự kiến

```text
.
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── configs/
│   ├── hog_svm.json
│   ├── hog_rf.json
│   └── sift_bovw_svm.json
├── data/                       # Không commit dữ liệu lớn
├── outputs/
│   ├── features/
│   ├── models/
│   ├── metrics/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── download_cifar10.py
│   │   └── cifar10.py
│   ├── features/
│   │   ├── hog.py
│   │   ├── sift.py
│   │   └── bovw.py
│   ├── models/
│   │   ├── svm.py
│   │   ├── decision_tree.py
│   │   └── random_forest.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── utils.py
├── scripts/
│   ├── extract_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiments.py
├── tests/
│   ├── test_hog.py
│   ├── test_sift.py
│   ├── test_svm.py
│   ├── test_tree.py
│   └── test_metrics.py
└── report/
    ├── main.tex
    ├── references.bib
    └── figures/
```

## 6. Ma trận thí nghiệm

### 6.1. Thí nghiệm tối thiểu

| ID | Đặc trưng | Mô hình | Mục đích |
|---|---|---|---|
| E0 | Pixel thô chuẩn hóa | Linear SVM | Baseline |
| E1 | HOG | Linear SVM | Pipeline chính |
| E2 | HOG | Random Forest | So sánh mô hình |
| E3 | SIFT + BoVW | Linear SVM | So sánh đặc trưng |

Nếu SIFT chưa đạt độ ổn định trong thời gian cho phép, E3 được chuyển thành thí nghiệm mở rộng và báo cáo rõ giới hạn; E0–E2 vẫn đủ tạo một project hoàn chỉnh.

### 6.2. Ablation ưu tiên

1. HOG với cell 4×4 so với 8×8.
2. Pixel thô so với HOG khi giữ nguyên Linear SVM.
3. Linear SVM so với Random Forest khi giữ nguyên HOG.
4. BoVW vocabulary size 64 so với 128 (nếu hoàn thành SIFT).

### 6.3. Quy tắc chọn mô hình

- Chọn siêu tham số theo macro F1 hoặc accuracy trên validation.
- Chỉ đánh giá test một lần sau khi chốt cấu hình.
- Ghi lại seed, cấu hình, commit/version code và thời gian chạy trong file kết quả JSON.

## 7. Lộ trình thực hiện

### Giai đoạn 1 — Khởi tạo và baseline

- Tạo cấu trúc repository, môi trường và quy ước cấu hình.
- Viết bộ tải CIFAR-10 trực tiếp từ file pickle chính thức.
- Cài đặt split validation phân tầng và metrics.
- Hoàn thành baseline pixel + Linear SVM trên tập con nhỏ.

**Đầu ra:** dữ liệu đọc đúng, pipeline train/evaluate chạy end-to-end.

### Giai đoạn 2 — HOG + SVM

- Cài đặt và unit test HOG.
- Cài đặt Linear SVM one-vs-rest.
- Kiểm tra gradient/loss trên dữ liệu nhỏ.
- Chạy tìm siêu tham số trên validation.

**Đầu ra:** pipeline chính HOG + SVM, loss curve và metrics validation.

### Giai đoạn 3 — Random Forest

- Cài đặt decision tree và kiểm thử trên dữ liệu toy.
- Cài đặt bootstrap, random feature selection và voting.
- Chạy HOG + Random Forest; ghi nhận tốc độ và chất lượng.

**Đầu ra:** kết quả so sánh SVM và Random Forest.

### Giai đoạn 4 — SIFT + BoVW

- Cài đặt từng thành phần SIFT, kiểm tra bằng ảnh biên/corner nhân tạo.
- Cài đặt k-means và BoVW.
- Chạy trên tập con trước, sau đó mở rộng nếu thời gian cho phép.

**Đầu ra:** kết quả SIFT + BoVW hoặc một phân tích có bằng chứng về giới hạn trên ảnh 32×32.

### Giai đoạn 5 — Thí nghiệm cuối và báo cáo

- Chốt cấu hình bằng validation.
- Chạy test, sinh confusion matrix, bảng số liệu và hình ảnh lỗi.
- Viết báo cáo LaTeX, kiểm tra số trang và khả năng tái lập.
- Hoàn thiện README với lệnh chạy từ đầu đến cuối.

**Đầu ra:** code, kết quả, PDF báo cáo không quá 10 trang.

## 8. Kế hoạch báo cáo LaTeX (tối đa 10 trang)

Phân bổ đề xuất:

1. **Tóm tắt** — 0,25 trang.
2. **Giới thiệu và mục tiêu** — 0,5 trang.
3. **Dữ liệu và tiền xử lý** — 0,75 trang.
4. **Cơ sở lý thuyết** — 2,5 đến 3 trang:
   - HOG;
   - SIFT/BoVW nếu đã triển khai;
   - SVM;
   - Random Forest.
5. **Thiết kế và cài đặt from scratch** — 1,5 trang.
6. **Thiết lập thí nghiệm** — 0,75 trang.
7. **Kết quả và thảo luận** — 2 đến 2,5 trang.
8. **Hạn chế, hướng phát triển và kết luận** — 0,5 đến 0,75 trang.
9. **Tài liệu tham khảo** — phần còn lại trong giới hạn 10 trang.

Nội dung cần ưu tiên:

- Giải thích trực giác trước, sau đó mới đưa công thức quan trọng.
- Nêu rõ phần nào được tự cài đặt và phần nào dùng thư viện.
- Không chép mã nguồn dài vào báo cáo; chỉ dùng pseudocode ngắn khi cần.
- Dùng một bảng kết quả chính, một confusion matrix, một hình minh họa HOG/SIFT và một nhóm ảnh dự đoán sai tiêu biểu.
- Mọi kết luận phải gắn với số liệu thí nghiệm.

## 9. Tiêu chí hoàn thành

Project được xem là hoàn thành khi:

- [ ] Tải, đọc và chia CIFAR-10 đúng, không rò rỉ dữ liệu test.
- [ ] Có ít nhất một bộ trích xuất đặc trưng truyền thống tự cài đặt: HOG.
- [ ] Có ít nhất hai mô hình/phương án so sánh hợp lệ; ưu tiên SVM và Random Forest tự cài đặt.
- [ ] Pipeline có thể chạy lại bằng lệnh rõ ràng trong README.
- [ ] Có unit test cho các thành phần toán học quan trọng.
- [ ] Có baseline và ít nhất hai thí nghiệm so sánh.
- [ ] Có accuracy, macro F1, confusion matrix và phân tích lỗi.
- [ ] Kết quả và cấu hình được lưu để tái lập.
- [ ] Báo cáo viết bằng LaTeX, tiếng Việt và PDF không quá 10 trang.
- [ ] Báo cáo giải thích được trực giác, công thức, cách cài đặt và giới hạn của phương pháp.

## 10. Rủi ro và phương án xử lý

| Rủi ro | Ảnh hưởng | Phương án |
|---|---|---|
| SIFT khó ổn định trên ảnh 32×32 | Ít keypoint, chất lượng BoVW thấp | Upscale có kiểm soát; giữ HOG là pipeline chính |
| Random Forest from scratch chạy chậm | Không kịp quét nhiều cấu hình | Giới hạn feature/threshold tại mỗi node; thử trên tập con trước |
| Kernel SVM tốn bộ nhớ | Không chạy được toàn bộ train set | Dùng Linear SVM cho kết quả chính; RBF chỉ là thí nghiệm phụ |
| SVM SGD không hội tụ | Kết quả dao động | Kiểm tra loss trên toy data; chuẩn hóa feature; giảm learning rate |
| Quá nhiều tổ hợp siêu tham số | Tốn thời gian | Dùng coarse search nhỏ, sau đó tinh chỉnh quanh cấu hình tốt nhất |
| Báo cáo vượt 10 trang | Không đạt yêu cầu hình thức | Chốt khung trang từ sớm; đưa chi tiết phụ vào README thay vì báo cáo |

## 11. Thứ tự ưu tiên khi thiếu thời gian

1. Pipeline dữ liệu + metrics đúng.
2. HOG from scratch.
3. Linear SVM from scratch.
4. Baseline và đánh giá/phân tích lỗi.
5. Random Forest from scratch.
6. Báo cáo LaTeX hoàn chỉnh.
7. SIFT + BoVW.
8. Kernel SVM hoặc SURF.

Nguyên tắc: ưu tiên một pipeline hoàn chỉnh, có kiểm thử và phân tích tốt hơn nhiều thuật toán chưa ổn định.

