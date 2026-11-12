# Hướng dẫn deploy model lên dashboard AquaFFIA (Deployment 2 / Deployment 3)

Tài liệu này dành cho người deploy **Deployment 2** và **Deployment 3** lên Jetson Orin Nano.
Bạn **không cần viết giao diện hay API**. Dashboard và server đã có sẵn. Bạn chỉ cần:

1. đặt file model vào thư mục `weights/`,
2. sửa **một file duy nhất** là `model.py`,
3. build container và kiểm tra bằng `check_contract.py`.

Sau đó trên dashboard `http://<jetson>:8001`, bấm nút **Deployment 2** (hoặc 3) ở góc phải là thấy kết quả model của bạn, trên đúng giao diện của Deployment 1.

> Kiến trúc tổng thể và đặc tả API đầy đủ nằm ở `~/aquaffia_edge_deployment/docs/DEPLOYMENT_API.md`.

---

## 1. Hệ thống hoạt động thế nào (đọc 2 phút)

```
Trình duyệt → http://<jetson>:8001  (aquaffia-edge-gateway: trang web + bật/tắt deployment)
                 │
                 ├─ Deployment 1: aquaffia-edge    cổng 8011
                 ├─ Deployment 2: aquaffia-edge-2  cổng 8012   ← thư mục ~/aquaffia_edge_deployment_2
                 └─ Deployment 3: aquaffia-edge-3  cổng 8013   ← thư mục ~/aquaffia_edge_deployment_3
```

- **Mỗi lúc chỉ một deployment chạy.** Khi bấm Deployment 2, gateway tắt deployment đang chạy rồi bật container của bạn. Nhờ vậy model của bạn có GPU, CPU và RAM một mình, và latency đo được là latency thật.
- Sau khi Jetson khởi động, Deployment 1 được bật mặc định.
- Cứ mỗi 2 giây, dashboard gọi `GET :8012/api/live/step?step=N`. Server trong container (`app.py`, **đã viết sẵn**) sẽ:
  1. đọc **đúng cửa sổ 2 giây thứ N** của video và audio mô phỏng, dùng chung file với Deployment 1 nên 3 model nhận cùng một input;
  2. gọi các hàm trong `model.py` của bạn và **tự đo latency** cho từng bước;
  3. trả về JSON đúng định dạng của Deployment 1, và dashboard hiển thị.

---

## 2. Thư mục của bạn trên Jetson

```
~/aquaffia_edge_deployment_2/        (Deployment 3: ~/aquaffia_edge_deployment_3/)
├── model.py              ★ BẠN SỬA FILE NÀY
├── weights/              ★ ĐẶT FILE MODEL VÀO ĐÂY (engine, checkpoint...)
├── .env                  tên hiển thị, cổng, tên container (chỉ sửa DEPLOYMENT_NAME nếu cần)
├── requirements.txt      thư viện Python cần cài thêm (để trống nếu không cần)
├── app.py                server: KHÔNG SỬA
├── trt_helper.py         hàm nạp TensorRT engine, dùng trong model.py
├── examples/model_aquaffia_reference.py   ví dụ hoàn chỉnh (model của Deployment 1)
├── check_contract.py     công cụ kiểm tra
├── contract/             dữ liệu chuẩn để kiểm tra
├── Dockerfile, docker-compose.yml        KHÔNG SỬA
```

Hiện thư mục đang chạy **model DEMO** (trả về kết quả ngẫu nhiên) để dashboard hoạt động trước khi bạn có model thật.

---

## 3. Các bước deploy

Mọi lệnh chạy trên Jetson: SSH vào `fptdanang@<jetson>` hoặc mở Terminal qua Remote Desktop.

### Bước 1. Chuẩn bị model

Khuyến nghị **TensorRT FP32, batch 1, shape cố định**, giống Deployment 1, để so latency công bằng.

```bash
# trên Jetson: build engine từ ONNX (batch 1, shape cố định)
/usr/src/tensorrt/bin/trtexec --onnx=my_model.onnx --saveEngine=my_model_fp32.engine
```

- Engine TensorRT **phải build ngay trên chính Jetson này**. Engine build trên máy khác không chạy được.
- Dùng được cả ONNX Runtime hoặc PyTorch (image đã có `torch`, `onnxruntime`), nhưng chậm hơn TensorRT.

### Bước 2. Đặt model vào `weights/`

```bash
cp my_model_fp32.engine ~/aquaffia_edge_deployment_2/weights/
```
Thư mục này được mount **chỉ đọc** vào container tại `/srv/deployment/weights`.

### Bước 3. Sửa `model.py`

Server gọi 4 hàm dưới đây. Mỗi cửa sổ 2 giây gọi 3 hàm cuối một lần.

| Hàm | Nhận vào | Trả về | Được đo vào |
|---|---|---|---|
| `load(weights_dir)` | đường dẫn thư mục `weights` | không trả gì | (gọi 1 lần lúc khởi động) |
| `preprocess_audio(waveform)` | `np.float32 [128000]`: 2 giây audio mono **64 kHz** | đầu vào audio của model (tuỳ bạn) | `latency.audio_ms` |
| `preprocess_video(frame_first, frame_last)` | 2 khung hình `np.uint8 [H, W, 3]` hệ màu **BGR**, độ phân giải gốc (2560×1440): khung đầu và khung cuối của cửa sổ | đầu vào video của model (tuỳ bạn) | `latency.video_ms` |
| `infer(audio_input, video_input)` | kết quả của 2 hàm trên | **4 logits** theo thứ tự `[none, strong, medium, weak]` | `latency.gpu_ms` |

Ngoài ra:
- Điền `INFO`: tên model, runtime, fold, accuracy. Thông tin này hiện trong `/api/status`.
- Nếu `infer()` trả về **xác suất** thay vì logits, đặt `OUTPUT_IS_PROBABILITY = True`.
- Model **chỉ dùng audio**: cho `preprocess_video` trả về `None` và bỏ qua tham số đó trong `infer`. Model chỉ dùng video thì làm ngược lại.
- Thứ tự lớp **bắt buộc** là `0 none, 1 strong, 2 medium, 3 weak`.

Ví dụ với TensorRT (xem đầy đủ trong `examples/model_aquaffia_reference.py`):

```python
import os, cv2, numpy as np
from trt_helper import TRTModule

INFO = {"name": "MyNet", "runtime": "NVIDIA TensorRT FP32", "fold": "Fold 00", "accuracy_baseline": "96.5%"}
_model = None

def load(weights_dir):
    global _model
    _model = TRTModule(os.path.join(weights_dir, "my_model_fp32.engine"))
    print(_model.describe())                      # in ra tên, shape các input/output

def preprocess_audio(waveform):                   # [128000] @ 64 kHz
    return my_logmel(waveform)                    # hàm tiền xử lý của bạn

def preprocess_video(frame_first, frame_last):    # BGR, độ phân giải gốc
    f0 = cv2.resize(frame_first, (224, 224))
    f1 = cv2.resize(frame_last, (224, 224))
    return my_normalize(f0, f1)

def infer(audio_input, video_input):
    return _model.infer(audio_input, video_input)  # -> [1, 4] logits
```

> Muốn cài thêm thư viện thì ghi vào `requirements.txt`. Lúc build Jetson cần có internet.

### Bước 4. Kiểm tra `.env`

Thường chỉ sửa `DEPLOYMENT_NAME`. **Không đổi** `DEPLOYMENT_ID`, `PORT`, `CONTAINER_NAME`, vì gateway dựa vào các giá trị này.

| Biến | Deployment 2 | Deployment 3 |
|---|---|---|
| `DEPLOYMENT_ID` | 2 | 3 |
| `PORT` | 8012 | 8013 |
| `CONTAINER_NAME` | aquaffia-edge-2 | aquaffia-edge-3 |
| `MODEL_MODULE` | model | model |

### Bước 5. Build và tạo container (chưa chạy)

```bash
cd ~/aquaffia_edge_deployment_2
docker compose up --no-start --build
```

Dùng `--no-start` vì **gateway mới là nơi bật/tắt container**. Đừng dùng `docker compose up -d` hay `docker start`, nếu không sẽ có 2 model chạy cùng lúc.

### Bước 6. Bật deployment của bạn

Cách 1: trên dashboard `http://<jetson>:8001`, bấm **Deployment 2**.
Cách 2: bằng lệnh:

```bash
curl -X POST http://127.0.0.1:8001/gateway/switch/2
# {"ok": true, "active": 2, "seconds": 7.9}
```

Nếu nhận `"ok": false` thì model khởi động lỗi và gateway đã tự quay về Deployment 1. Xem log để biết nguyên nhân:

```bash
docker logs aquaffia-edge-2
```

### Bước 7. Kiểm tra hợp đồng API

Khi deployment của bạn **đang được bật**:

```bash
cd ~/aquaffia_edge_deployment_2
python3 check_contract.py --url http://127.0.0.1:8012
```

Phải ra `RESULT: PASS`. Script kiểm tra:
- `/api/status` trả về `online` và cho phép dashboard gọi sang (CORS);
- JSON của `/api/live/step` có **đủ trường và đúng kiểu** như Deployment 1;
- giá trị hợp lệ: lớp nằm trong 0..3, tổng xác suất là 100, `confidence`, `feeder_action` và `stream_info` đúng;
- **audio ở step 10 giống hệt Deployment 1**, chứng tỏ container của bạn nhận đúng input.

### Bước 8. Xem trên dashboard

Mở `http://<jetson>:8001/?d=2` (hoặc bấm nút Deployment 2). Bấm **Deployment 1** để quay về mặc định.

---

## 4. Cập nhật model sau này

```bash
cd ~/aquaffia_edge_deployment_2
# sửa model.py hoặc thay file trong weights/
docker compose up --no-start --build      # build lại; container đang chạy sẽ bị dừng
curl -X POST http://127.0.0.1:8001/gateway/switch/2
python3 check_contract.py --url http://127.0.0.1:8012
```

Nếu chỉ thay file trong `weights/` mà không sửa code: `docker restart aquaffia-edge-2`, khi deployment của bạn đang được bật.

---

## 5. Quy định bắt buộc

| Quy định | Lý do |
|---|---|
| Chỉ sửa `model.py`, `weights/`, `requirements.txt`, `DEPLOYMENT_NAME` | `app.py` đảm bảo JSON và input giống hệt Deployment 1 |
| Không tự `docker start` hoặc `docker compose up -d` | Gateway đảm bảo mỗi lúc chỉ một model chạy |
| Model phải sẵn sàng trong **120 giây** sau khi bật | Quá thời gian, gateway coi như lỗi và quay về Deployment 1 |
| RAM dùng không quá **khoảng 2.5 GB** | Jetson có 7.6 GB dùng chung CPU và GPU. Deployment 1 dùng khoảng 1.3 GB |
| Thứ tự lớp `[none, strong, medium, weak]` | Dashboard hiển thị theo thứ tự này |
| Batch 1, khuyến nghị TensorRT FP32 | So latency công bằng với Deployment 1 |
| Trước khi bàn giao, chuyển dashboard về **Deployment 1** | Deployment 1 là mặc định |

---

## 6. Lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| Nút Deployment 2 bị mờ, ghi "Not deployed yet" | Chưa tạo container: chạy **Bước 5** |
| `"ok": false, "error": "Deployment 2 failed to start"` | `model.py` lỗi, hoặc thiếu file trong `weights/`. Xem `docker logs aquaffia-edge-2` |
| `ModuleNotFoundError` | Thiếu thư viện: thêm vào `requirements.txt` rồi build lại |
| `Failed to deserialize TensorRT engine` | Engine không build trên Jetson này, hoặc khác phiên bản TensorRT. Build lại bằng `trtexec` ngay trên Jetson |
| `has a dynamic shape` | Build engine với shape cố định (batch 1) |
| `check_contract`: `audio window at step 10 is identical` FAIL | Bạn đã sửa `app.py`, hoặc đổi `SIM_SOURCE` trong `.env`. Khôi phục lại |
| `check_contract`: không kết nối được | Deployment của bạn chưa được bật: chạy **Bước 6** trước |
| Latency cao bất thường | Chạy `sudo jetson_clocks` (mất sau mỗi lần reboot), kiểm tra lại là đang dùng TensorRT |

---

## 7. Lệnh hữu ích

```bash
curl -s http://127.0.0.1:8001/gateway/state | python3 -m json.tool   # deployment nào đang chạy
curl -X POST http://127.0.0.1:8001/gateway/switch/1                  # quay về Deployment 1
docker logs -f aquaffia-edge-2                                        # log của bạn
docker stats --no-stream                                              # RAM/CPU các container
curl -s "http://127.0.0.1:8012/api/live/step?step=0" | python3 -m json.tool   # thử 1 bước
```
