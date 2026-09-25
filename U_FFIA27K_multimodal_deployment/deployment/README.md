# JETSON ORIN NANO DEPLOYMENT PIPELINE

Thư mục này chứa toàn bộ mã nguồn, cấu hình và kịch bản phục vụ cho việc tối ưu hóa, đánh giá và triển khai mô hình Phân loại độ thèm ăn của cá (Fish Feeding Intensity) lên thiết bị biên **NVIDIA Jetson Orin Nano**.

---

## 1. Cấu trúc thư mục `deployment/`

```
U_FFIA27K_multimodal_deployment/
├── deployment/
│   ├── README.md               # Tài liệu hướng dẫn quy trình triển khai
│   ├── __init__.py             # Python package marker
│   ├── export_onnx.py          # [Giai đoạn 1] Export PyTorch checkpoint sang ONNX & kiểm chứng số học
│   ├── build_tensorrt.sh       # [Giai đoạn 1] Script biên dịch engine TensorRT FP16 trên Jetson
│   ├── prepare_dataset.py      # [Giai đoạn 2] Trích xuất & đóng gói tập mẫu 200 samples cân bằng
│   ├── eval_tensorrt.py        # [Giai đoạn 2 & 3] Đánh giá độ chính xác (CV) & đo latency trên Jetson
│   └── web_dashboard/          # [Giai đoạn 4] Ứng dụng Web Dashboard thời gian thực (FastAPI + Web UI)
│       ├── app.py
│       ├── static/
│       └── templates/
├── checkpoint/                 # Trọng số mô hình huấn luyện (Fold 00 đến Fold 04)
├── weights/                    # Thư mục chứa model ONNX và TensorRT engine (.gitignore)
├── samples/                    # Thư mục chứa dữ liệu mẫu kiểm thử (.gitignore)
```

---

## 2. Quy trình thực hiện từng giai đoạn

### Giai đoạn 1: Chuẩn bị & Tối ưu hóa mô hình sang TensorRT
1. **Export sang ONNX và kiểm tra đối soát trên PC:**
   ```bash
   python deployment/export_onnx.py --checkpoint checkpoint/multimodal_model/fold_00/multimodal_best.pt --output_dir weights/
   ```
   *Đầu ra:* `weights/multimodal_core_sim.onnx` (15.26 MB), kiểm tra độ khớp 100% với PyTorch.

2. **Biên dịch engine TensorRT FP16 trên Jetson Orin Nano:**
   ```bash
   bash deployment/build_tensorrt.sh weights/multimodal_core_sim.onnx weights/multimodal_core_fp16.engine
   ```
   *Đầu ra:* `weights/multimodal_core_fp16.engine` (12 MB), đạt 12.59 ms GPU latency (~79 FPS).

---

### Giai đoạn 2: Chuẩn bị dữ liệu mẫu & Xác thực Cross-Validation
1. **Trích xuất 200 mẫu test cân bằng 4 lớp:**
   ```bash
   python deployment/prepare_dataset.py --samples_per_class 50
   ```
   *Đầu ra:* `samples/test_samples.csv` và `samples/test_samples_200.zip`.

2. **Chạy đánh giá độ chính xác trên Jetson:**
   ```bash
   python3 deployment/eval_tensorrt.py --engine weights/multimodal_core_fp16.engine --samples_dir samples/
   ```
   *Đầu ra:* Ma trận nhầm lẫn (Confusion Matrix 4x4) và chỉ số Accuracy, Precision, Recall, F1 so sánh với kết quả gốc của Fold 00 (~97.10%).

---

### Giai đoạn 3: Đo đạc tốc độ suy luận (Latency Benchmark)
Script `eval_tensorrt.py` tự động đo đạc chi tiết:
- Thời gian trích xuất Mel-spectrogram âm thanh (`t_audio`).
- Thời gian trích xuất & chuẩn hóa frame video (`t_video`).
- Thời gian GPU inference qua TensorRT (`t_gpu`).
- Tổng thời gian toàn trình (`t_total`) và FPS thực tế.

---

### Giai đoạn 4: Vận hành Edge Web Dashboard
Khởi chạy ứng dụng Web giám sát thời gian thực trên Jetson:
```bash
python3 deployment/web_dashboard/app.py --port 8000
```
- Truy cập qua cáp Type-C: `http://192.168.55.1:8000`
- Truy cập qua Wi-Fi nội bộ: `http://<Jetson_IP>:8000`
