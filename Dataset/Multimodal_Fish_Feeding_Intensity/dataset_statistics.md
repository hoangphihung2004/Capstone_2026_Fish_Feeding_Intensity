# Thống kê bộ dữ liệu Multimodal Fish Feeding Intensity

Đường dẫn dataset:

```text
D:\Fish_Feeding_Intensity\Dataset\Multimodal_Fish_Feeding_Intensity
```

## 1. Tổng quan dataset

Bộ dữ liệu gồm 3 dạng dữ liệu chính:

| Modality | Kiểu file | Số file |
|---|---:|---:|
| Audio | `.wav` | 7,089 |
| Image | `.jpg` | 7,089 |
| Wave | `.csv` | 7,089 |
| **Tổng** |  | **21,267** |

Tổng dung lượng toàn bộ dataset khoảng **10.64 GB**.

Mỗi sample đa phương thức gồm 3 file tương ứng:

```text
audio_xxxx.wav
image_xxxx.jpg
wave_xxxx.csv
```

Do đó, tổng số sample đa phương thức là **7,089 samples**.

## 2. Phân bố sample theo mức feeding intensity

Dataset hiện tại có 3 mức nhãn:

- `None`
- `Weak`
- `Strong`

Không thấy thư mục hoặc mức nhãn `Medium` trong dataset hiện tại.

| Mức nhãn | Số sample | Tỷ lệ | Audio | Image | Wave |
|---|---:|---:|---:|---:|---:|
| None | 2,327 | 32.83% | 2,327 | 2,327 | 2,327 |
| Weak | 2,353 | 33.19% | 2,353 | 2,353 | 2,353 |
| Strong | 2,409 | 33.98% | 2,409 | 2,409 | 2,409 |
| **Tổng** | **7,089** | **100%** | **7,089** | **7,089** | **7,089** |

Nhìn chung, dữ liệu tương đối cân bằng giữa 3 lớp. Lớp `Strong` có số lượng lớn nhất với 2,409 sample, trong khi lớp `None` có số lượng nhỏ nhất với 2,327 sample.

## 3. Thông tin Audio

Toàn bộ file audio có định dạng `.wav`.

| Thuộc tính | Giá trị |
|---|---:|
| Sample rate | 48,000 Hz |
| Số kênh | 2 channels |
| Bit depth | 32-bit |
| Thời lượng mỗi file | 1.0 giây |
| Tổng số file audio | 7,089 |

Phân bố file audio theo nhãn:

| Nhãn | Số file | Dung lượng |
|---|---:|---:|
| None | 2,327 | 852.27 MB |
| Weak | 2,353 | 861.79 MB |
| Strong | 2,409 | 882.30 MB |
| **Tổng** | **7,089** | **2,596.36 MB** |

## 4. Thông tin Wave / Sensor CSV

Toàn bộ file wave có định dạng `.csv`.

Mỗi file CSV gồm 1 dòng header và 200 dòng dữ liệu. Vì mỗi sample đồng bộ với audio dài 1 giây, tần số lấy mẫu của dữ liệu wave có thể được xem là khoảng **200 Hz**.

Các cột dữ liệu trong file CSV:

```text
acceleration_x, acceleration_y, acceleration_z,
palstance_x, palstance_y, palstance_z,
angle_x, angle_y, angle_z
```

Ý nghĩa nhóm đặc trưng:

| Nhóm đặc trưng | Các cột |
|---|---|
| Acceleration | `acceleration_x`, `acceleration_y`, `acceleration_z` |
| Palstance | `palstance_x`, `palstance_y`, `palstance_z` |
| Angle | `angle_x`, `angle_y`, `angle_z` |

Phân bố file wave theo nhãn:

| Nhãn | Số file | Dung lượng |
|---|---:|---:|
| None | 2,327 | 27.52 MB |
| Weak | 2,353 | 27.79 MB |
| Strong | 2,409 | 28.69 MB |
| **Tổng** | **7,089** | **84.00 MB** |

## 5. Thông tin Image

Toàn bộ file ảnh có định dạng `.jpg`.

| Thuộc tính | Giá trị |
|---|---:|
| Kiểu file | JPG |
| Tổng số file ảnh | 7,089 |
| Kích thước ảnh | Không cố định hoàn toàn |

Các kích thước ảnh xuất hiện trong dataset:

| Kích thước ảnh | None | Weak | Strong | Tổng |
|---|---:|---:|---:|---:|
| 2800x2080 | 9 | 405 | 0 | 414 |
| 2900x2110 | 121 | 69 | 120 | 310 |
| 2950x2110 | 576 | 519 | 324 | 1,419 |
| 3000x2110 | 947 | 785 | 884 | 2,616 |
| 3000x2130 | 674 | 575 | 1,081 | 2,330 |
| **Tổng** | **2,327** | **2,353** | **2,409** | **7,089** |

Phân bố file ảnh theo nhãn:

| Nhãn | Số file | Dung lượng |
|---|---:|---:|
| None | 2,327 | 2,546.39 MB |
| Weak | 2,353 | 2,647.37 MB |
| Strong | 2,409 | 3,018.28 MB |
| **Tổng** | **7,089** | **8,212.04 MB** |

## 6. Kiểm tra tính khớp giữa các modality

Trong từng nhãn, số lượng ID giữa 3 modality là khớp nhau hoàn toàn.

| Nhãn | Audio IDs | Image IDs | Wave IDs | Sample đủ 3 modality |
|---|---:|---:|---:|---:|
| None | 2,327 | 2,327 | 2,327 | 2,327 |
| Weak | 2,353 | 2,353 | 2,353 | 2,353 |
| Strong | 2,409 | 2,409 | 2,409 | 2,409 |
| **Tổng** | **7,089** | **7,089** | **7,089** | **7,089** |

Điều này cho thấy dataset có cấu trúc đa phương thức 1-1-1: mỗi sample có đủ audio, image và wave tương ứng.

## 7. Kết luận

Bộ dữ liệu `Multimodal_Fish_Feeding_Intensity` gồm **7,089 sample đa phương thức**, chia thành 3 mức feeding intensity: `None`, `Weak`, và `Strong`. Mỗi sample gồm 1 file audio WAV dài 1 giây ở sample rate 48 kHz, 1 ảnh JPG, và 1 file CSV cảm biến gồm 200 dòng dữ liệu với 9 đặc trưng.

Dataset có phân bố lớp tương đối cân bằng, phù hợp cho các bài toán phân loại cường độ ăn của cá dựa trên dữ liệu đa phương thức.
