# CIFAR-10: AlexNet, VGG11, ResNet18 và ViT

Bốn mạng được cài đặt bằng các lớp cơ bản của PyTorch và huấn luyện từ trọng số ngẫu nhiên, không dùng trọng số huấn luyện sẵn.

Các kiến trúc được điều chỉnh cho ảnh 32×32: AlexNet dùng năm lớp tích chập với GroupNorm và bộ phân lớp hai lớp; VGG11-BN dùng tám lớp tích chập với số kênh giảm; ResNet18 dùng tám block phần dư, lớp tích chập đầu 3×3 và 32 kênh ban đầu; ViT tiny dùng patch 4×4, vector 192 chiều, sáu block và ba attention head.

Chế độ: **thí nghiệm**. Seed: **42**. Epoch tối đa: **200**. Số ảnh huấn luyện/kiểm định/kiểm tra: **45000/5000/10000**. Thiết bị: **mps**.

Tập kiểm định được tách phân tầng từ 50.000 ảnh huấn luyện chính thức. Bộ trọng số tốt nhất được chọn theo accuracy kiểm định, sau đó mới đánh giá trên tập kiểm tra. Chuẩn hóa dùng thống kê của tập huấn luyện; không dùng tăng cường dữ liệu.

| Mô hình | Số tham số | Epoch chọn | Accuracy kiểm định | Accuracy kiểm tra | Macro-F1 kiểm tra | Huấn luyện (giờ) |
|---|---:|---:|---:|---:|---:|---:|
| AlexNet (GroupNorm) | 2,786,890 | 193 | 0.8518 | 0.8387 | 0.8381 | 3.48 |
| VGG11-BN | 2,310,186 | 130 | 0.8120 | 0.7952 | 0.7944 | 2.19 |
| ResNet18 | 2,797,610 | 184 | 0.8262 | 0.8145 | 0.8141 | 6.51 |
| ViT tiny | 2,693,578 | 180 | 0.6888 | 0.6821 | 0.6803 | 5.18 |

Kết quả được đánh giá trên 10.000 ảnh kiểm tra; accuracy và macro-F1 được trình bày trên thang [0,1].

## So sánh với phương pháp truyền thống

![Accuracy và macro-F1 trên CIFAR-10](assets/figures/deep_vs_traditional_200.png)

Biểu đồ đối chiếu các mạng học sâu với sáu cấu hình truyền thống. SVM và các mạng chạy 200 epoch; Random Forest dùng 200 cây. Các thí nghiệm dùng cùng số ảnh và seed 42, nhưng khác cách tối ưu và chi phí tính toán.
