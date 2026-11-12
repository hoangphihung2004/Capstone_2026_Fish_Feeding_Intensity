# AquaFFIA multi-deployment: kiến trúc và đặc tả API

Tài liệu kỹ thuật cho hệ thống chạy **3 deployment** (Deployment 1, 2, 3) trên cùng một Jetson Orin Nano và hiển thị trên **một dashboard duy nhất**.
Hướng dẫn từng bước cho người deploy Deployment 2/3 nằm ở `deployment_template/README.md`.

---

## 1. Kiến trúc

```
                     Trình duyệt → http://<jetson>:8001
                                        │
             ┌──────────────────────────▼──────────────────────────┐
             │ aquaffia-edge-gateway (8001), luôn bật, KHÔNG chạy model │
             │  • trang dashboard (web_dashboard/), /static, video  │
             │  • bật/tắt deployment, đảm bảo mỗi lúc chỉ 1 cái     │
             └──────┬───────────────────┬───────────────────┬──────┘
                    │                   │                   │
         Deployment 1          Deployment 2          Deployment 3
         aquaffia-edge         aquaffia-edge-2       aquaffia-edge-3
         cổng 8011             cổng 8012             cổng 8013
         ~/aquaffia_edge_      ~/aquaffia_edge_      ~/aquaffia_edge_
           deployment            deployment_2          deployment_3
                    └──── cùng đọc samples/continuous_simulation (chỉ đọc) ────┘

deploy_jetson_web (8000): dự án khác, độc lập
```

| Container | Cổng | Thư mục / nguồn | Restart | Vai trò |
|---|---|---|---|---|
| `aquaffia-edge-gateway` | 8001 | `~/aquaffia_edge_deployment/deployment_gateway` | `unless-stopped` | Trang web, bật/tắt deployment |
| `aquaffia-edge` | 8011 | `~/aquaffia_edge_deployment/docker` | do gateway điều khiển | Deployment 1 (model AquaFFIA) |
| `aquaffia-edge-2` | 8012 | `~/aquaffia_edge_deployment_2` | `no` | Deployment 2 |
| `aquaffia-edge-3` | 8013 | `~/aquaffia_edge_deployment_3` | `no` | Deployment 3 |

Nguyên tắc:
- **Mỗi lúc chỉ một model chạy.** Model đang được xem có GPU, CPU và RAM một mình, nên latency không bị ảnh hưởng bởi các deployment khác.
- **Sau khi khởi động, Deployment 1 là mặc định.** Gateway bật D1 và tắt D2/D3.
- Code của Deployment 1 (`web_dashboard/app.py`) **không thay đổi**. Chỉ cổng mặc định chuyển từ 8001 sang 8011, và giao diện thêm nút chuyển.
- Giao diện chỉ có **một bản** (`web_dashboard/templates/index.html`). Gateway mount file này chỉ đọc.

---

## 2. Luồng hoạt động

### 2.1. Khi Jetson khởi động
1. Docker bật `aquaffia-edge-gateway`.
2. Gateway gọi `switch(1)`: tắt D2/D3 nếu đang chạy, bật D1, rồi chờ `:8011/api/status` sẵn sàng.
3. Mở `http://<jetson>:8001` là thấy Deployment 1.

### 2.2. Một nhịp 2 giây (đang xem Deployment N)
```
Trình duyệt ── GET :801N/api/live/step?step=S ──► container của deployment N
                                                    1. đọc cửa sổ 2 giây thứ S (audio 64 kHz + khung hình đầu/cuối)
                                                    2. tiền xử lý audio → video → suy luận (đo latency từng bước)
                                                    3. softmax, feeder_action, sóng âm, stream_info
Trình duyệt ◄── JSON (định dạng chung) ───────────
updateLiveFarmUI(data): vẽ kết quả trên cùng một giao diện; video đồng bộ theo stream_info.time_sec
```
Trình duyệt giữ bộ đếm `step` chung (`stream_info.next_step`). Vì vậy chuyển deployment thì video chạy tiếp, không quay lại đầu, và cùng một step thì mọi deployment nhận **cùng một input**.

### 2.3. Khi bấm chuyển sang Deployment N
```
Trình duyệt ── POST /gateway/switch/N ──► gateway
                                            1. tắt mọi deployment khác đang chạy
                                            2. bật container của N
                                            3. chờ :801N/api/status = online (model đã nạp và warmup)
Trình duyệt ◄── {"ok": true, "active": N, "seconds": ...}
              tiếp tục gọi :801N/api/live/step?step=S
```
- Trong lúc chuyển, giao diện hiện "Starting Deployment N…" và tạm dừng gọi suy luận.
- Nếu N không sẵn sàng sau 120 giây, hoặc container bị dừng giữa chừng, gateway tắt N, **bật lại Deployment 1** và trả về `{"ok": false, "active": 1, "error": "..."}`.
- Mỗi lúc chỉ xử lý một lệnh chuyển. Lệnh thứ hai gửi tới trong lúc đó nhận HTTP 409.
- Mọi người xem dùng chung **một deployment đang hoạt động**. Giao diện của người khác tự theo sau tối đa 3 giây.

### 2.4. Trạng thái container

| Đang xem | gateway | D1 | D2 | D3 |
|---|---|---|---|---|
| Deployment 1 (mặc định) | ● | ● | ○ | ○ |
| Deployment 2 | ● | ○ | ● | ○ |
| Deployment 3 | ● | ○ | ○ | ● |

---

## 3. API của deployment (hợp đồng chung D1, D2, D3)

Mọi deployment **phải** có các endpoint dưới đây và trả về JSON đúng định dạng. Bộ khung `deployment_template/app.py` đã cài đặt sẵn, người deploy chỉ sửa `model.py`.

### 3.1. `GET /api/live/step?step=N`

Chạy suy luận trên cửa sổ 2 giây thứ `N` của luồng mô phỏng (giây `2N` → `2N+2`, `N` từ 0 tới 298).
Không truyền `step` thì chạy cửa sổ kế tiếp của luồng.

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `predicted_class` | int | 0 none, 1 strong, 2 medium, 3 weak |
| `class_name` | str | `none` / `strong` / `medium` / `weak` |
| `confidence` | float | Xác suất (%) của lớp dự đoán |
| `probabilities.{none,strong,medium,weak}` | float | Xác suất từng lớp (%), tổng bằng 100 |
| `latency.audio_ms` | float | Thời gian tiền xử lý audio (ms) |
| `latency.video_ms` | float | Thời gian tiền xử lý video (ms) |
| `latency.gpu_ms` | float | Thời gian chạy model chính (ms) |
| `latency.total_ms` | float | Tổng thời gian, tính từ khi đủ dữ liệu 2 giây tới khi có kết quả |
| `latency.fps`, `latency.pipeline_fps` | float | `1000 / gpu_ms`, `1000 / total_ms` |
| `feeder_action.relay` | str | `ON` nếu lớp strong hoặc medium, ngược lại `OFF` |
| `feeder_action.status`, `.reason`, `.color` | str | Nội dung hiển thị cho máy cho ăn |
| `audio_telemetry.waveform` | list[600][3] | Sóng âm `[min, max, rms]` đã chuẩn hoá, dùng để vẽ |
| `audio_telemetry.rms_db`, `.peak_amp` | float | Mức âm của cửa sổ |
| `audio_telemetry.window_sec`, `.sample_rate` | float, int | 2.0, 64000 |
| `frame_base64` | null | Dự phòng, hiện không dùng |
| `stream_info.step`, `.next_step`, `.total_steps` | int | Bước vừa chạy, bước kế tiếp, tổng số bước (299) |
| `stream_info.time_sec`, `.time_str` | float, str | Vị trí trong video (giây), ví dụ `00:20 / 09:58` |
| `stream_info.window_sec`, `.video_url`, `.audio_url`, `.session` | | Thông tin luồng mô phỏng |

Mẫu đầy đủ: `deployment_template/contract/reference_step_10.json` (phản hồi của Deployment 1 ở step 10).

### 3.2. `GET /api/status`
Trả về `{"status": "online", ...}` **chỉ khi model đã nạp và warmup xong**. Gateway dựa vào endpoint này để biết deployment đã sẵn sàng.

### 3.3. `GET /api/live/info`, `POST /api/live/reset`
Thông tin luồng mô phỏng, và đưa luồng về bước 0. Giống Deployment 1.

### 3.4. CORS
Phải cho phép trang ở cổng 8001 gọi sang. Deployment 1 cho phép mọi nguồn. Bộ khung cho phép `http(s)://<host>:8001`.

---

## 4. API của gateway (cổng 8001)

| Endpoint | Mô tả |
|---|---|
| `GET /` | Trang dashboard |
| `GET /static/...`, `GET /samples/continuous_simulation/...` | File tĩnh và video (hỗ trợ Range) |
| `GET /gateway/state` | Trạng thái (xem dưới) |
| `POST /gateway/switch/{1\|2\|3}` | Chuyển deployment, chờ tới khi sẵn sàng rồi mới trả về |

```json
GET /gateway/state
{
  "active": 1, "switching": false, "target": null, "message": "",
  "deployments": [
    {"id": 1, "name": "Deployment 1", "container": "aquaffia-edge",   "port": 8011, "status": "running", "ready": true},
    {"id": 2, "name": "Deployment 2", "container": "aquaffia-edge-2", "port": 8012, "status": "created", "ready": false},
    {"id": 3, "name": "Deployment 3", "container": "aquaffia-edge-3", "port": 8013, "status": "missing", "ready": false}
  ]
}
```
`status` lấy từ Docker: `running`, `exited`, `created` (đã tạo, chưa chạy), hoặc `missing` (chưa deploy).

| Mã trả về của `POST /gateway/switch/{id}` | Ý nghĩa |
|---|---|
| 200 `{"ok": true, "active": id, "seconds": s}` | Đã chuyển xong |
| 200 `{"ok": false, "active": 1, "error": ...}` | Deployment `id` không khởi động được, đã quay về Deployment 1 |
| 404 | `id` không tồn tại |
| 409 | `id` chưa deploy, hoặc đang có lệnh chuyển khác |

Bảo mật: gateway điều khiển Docker qua `/var/run/docker.sock` nhưng **chỉ start/stop đúng 3 container** khai báo trong `DEPLOYMENTS` của `gateway.py`.

---

## 5. Giao diện

- Nút chuyển ở góc phải header: `Deployment 1 | Deployment 2 | Deployment 3`.
  - Chấm **xanh**: đang chạy và sẵn sàng. **Vàng**: đang khởi động. **Xám**: đang tắt.
  - Nút mờ: chưa deploy.
- Có thể mở thẳng một deployment bằng `?d=2`.
- Nút chỉ hiện khi trang được phục vụ qua gateway (8001). Mở trực tiếp `:8011` thì trang vẫn hoạt động như trước, chỉ không có nút chuyển.
- Ngoài phần trên, toàn bộ bố cục, số liệu và hàm `updateLiveFarmUI()` giữ nguyên.

---

## 6. Vận hành

```bash
# Gateway (một lần; sau đó tự bật cùng Jetson)
cd ~/aquaffia_edge_deployment/deployment_gateway && docker compose up -d --build

# Deployment 1: build lại image sau khi đổi code (container do gateway bật/tắt)
cd ~/aquaffia_edge_deployment && docker/run_web.sh build && docker compose -f docker/docker-compose.yml up --no-start web
curl -X POST http://127.0.0.1:8001/gateway/switch/1

# Deployment 2/3: xem deployment_template/README.md
cd ~/aquaffia_edge_deployment_2 && docker compose up --no-start --build

# Chuyển và xem trạng thái
curl -X POST http://127.0.0.1:8001/gateway/switch/2
curl -s http://127.0.0.1:8001/gateway/state | python3 -m json.tool
```

**Benchmark** (`docker/run_benchmark.sh`): chuyển về Deployment 1 trước khi chạy, để D2/D3 tắt và không chiếm RAM. Script tự tắt D1 trong lúc chạy rồi bật lại. Gateway **không** tự bật lại D1, nên không tranh GPU với benchmark.

**Khôi phục cấu hình cũ** (một container ở cổng 8001, không có gateway):
```bash
cd ~/aquaffia_edge_deployment/deployment_gateway && docker compose down
cd ~/aquaffia_edge_deployment && AQUAFFIA_PORT=8001 docker compose -f docker/docker-compose.yml up -d web
```

---

## 7. Kết quả kiểm thử khi bàn giao (Jetson Orin Nano, `jetson_clocks` bật)

| Kiểm thử | Kết quả |
|---|---|
| Latency Deployment 1 trước khi thay đổi (trung bình 35 nhịp) | total **18.49 ms** (audio 5.04 / video 3.01 / GPU 10.41) |
| Latency Deployment 1 sau khi thay đổi | total **18.60 ms** và 19.35 ms (2 lần đo), trong ±1 ms |
| Model của Deployment 1 chạy qua bộ khung (`examples/model_aquaffia_reference.py`) | **51/51 bước trùng hoàn toàn** với Deployment 1. Latency 18.82 ms |
| `check_contract.py` (model tham chiếu / model demo) | PASS 16/16, PASS 14/14 |
| Thời gian chuyển | D1→D2 7.9 s, D1→D3 7.4 s, D3→D2 17.0 s, D2→D1 14.7 s |
| RAM | gateway 33 MB; mỗi deployment khoảng 1.25–1.33 GB; luôn chỉ một model trong RAM |
| Bấm chuyển khi đang chuyển | HTTP 409, không bị kẹt |
| Deployment lỗi khi khởi động | Tự quay về Deployment 1 sau 7.7 s |
| Benchmark fold 04 sau khi thay đổi | **96.53% (5224/5412)**, confusion matrix trùng từng ô với trước khi thay đổi |
| Reboot khi Deployment 2 đang chạy | Sau khi khởi động: gateway tự bật Deployment 1 (sẵn sàng sau 6.0 s), D2/D3 tắt |

Lưu ý về tính lặp lại của dashboard: luồng mô phỏng dùng video H.264. Lệnh nhảy khung hình (`cv2.CAP_PROP_POS_FRAMES`) trên H.264 có thể rơi vào khung hình khác nhau tuỳ vị trí đang đọc. Vì vậy cùng một step, dự đoán trên dashboard có thể khác giữa hai lần chạy nếu trước đó luồng đứng ở vị trí khác. Hiện tượng này có sẵn trong Deployment 1, không liên quan tới bộ khung, và không ảnh hưởng benchmark (benchmark đọc trực tiếp từng clip gốc).
