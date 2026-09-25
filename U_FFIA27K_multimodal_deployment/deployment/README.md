# JETSON ORIN NANO DEPLOYMENT PIPELINE

Thư mục `deployment/` được tổ chức thành các module chuyên biệt, độc lập theo đúng từng chức năng kỹ thuật trong quy trình triển khai:

---

## 1. Cấu trúc module chuyên biệt

```
U_FFIA27K_multimodal_deployment/
├── deployment/
│   ├── model_converter/        # [MODULE 1] Chuyển đổi mô hình: PyTorch -> ONNX -> TensorRT Engine
│   │   ├── export_onnx.py      # Export mô hình PyTorch sang ONNX, kiểm chứng số học
│   │   └── build_tensorrt.sh   # Biên dịch engine TensorRT FP16 trên Jetson Orin Nano
│   │
│   ├── data_preparation/      # [MODULE 2] Chuẩn bị dữ liệu: Trích xuất & đóng gói tập mẫu test
│   │   └── prepare_dataset.py  # Trích xuất 200 mẫu test cân bằng 4 lớp kèm manifest
│   │
│   ├── benchmarking/          # [MODULE 3] Đánh giá: Cross-Validation & đo Latency trên Jetson
│   │   └── eval_tensorrt.py    # Đo Accuracy, Confusion Matrix, Latency breakdown (ms), FPS
│   │
│   ├── web_dashboard/         # [MODULE 4] Ứng dụng Web: Giao diện giám sát thời gian thực
│   │   ├── app.py              # Backend FastAPI chạy trên Jetson port 8000
│   │   ├── static/             # Tệp CSS, JS, Icon
│   │   └── templates/          # Giao diện HTML5 hiện đại
│   │
│   ├── README.md               # Tài liệu tổng quan quy trình triển khai
│   └── __init__.py             # Python package marker
│
├── checkpoint/                 # Trọng số mô hình huấn luyện gốc
├── weights/                    # Chứa file ONNX, TensorRT engine (.gitignore)
└── samples/                    # Chứa dữ liệu mẫu test (.gitignore)
```

---

## 2. Hướng dẫn sử dụng từng module

### Module 1: `model_converter` (Chuyển đổi sang ONNX + TensorRT)
1. **Chuyển đổi PyTorch sang ONNX và kiểm tra đối soát trên PC:**
   ```bash
   python deployment/model_converter/export_onnx.py \
       --checkpoint checkpoint/multimodal_model/fold_00/multimodal_best.pt \
       --output_dir weights/
   ```
2. **Biên dịch engine TensorRT FP16 trên Jetson:**
   ```bash
   bash deployment/model_converter/build_tensorrt.sh \
       weights/multimodal_core_sim.onnx \
       weights/multimodal_core_fp16.engine
   ```

---

### Module 2: `data_preparation` (Chuẩn bị mẫu kiểm thử)
Trích xuất tập mẫu đại diện cân bằng đủ 4 lớp:
```bash
python deployment/data_preparation/prepare_dataset.py --samples_per_class 50
```

---

### Module 3: `benchmarking` (Đo Latency & Đánh giá Cross-Validation)
Chạy script kiểm tra độ chính xác và tốc độ trên Jetson:
```bash
python3 deployment/benchmarking/eval_tensorrt.py \
    --engine weights/multimodal_core_fp16.engine \
    --samples_dir samples/
```

---

### Module 4: `web_dashboard` (Khởi chạy ứng dụng Web)
Khởi chạy giao diện web trên Jetson để các máy trong mạng cùng truy cập:
```bash
python3 deployment/web_dashboard/app.py --port 8000
```
- Truy cập Type-C: `http://192.168.55.1:8000`
- Truy cập Wi-Fi: `http://<Jetson_IP>:8000`
