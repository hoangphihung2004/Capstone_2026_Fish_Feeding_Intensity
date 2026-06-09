# U-FFIA Data Split

File `data_split.py` dùng để chia bộ dữ liệu multimodal U-FFIA thành `train`, `test`, `val` theo đúng logic trong source gốc `U-FFIA`.

## 1. Cấu trúc dữ liệu yêu cầu

Dataset mặc định nằm ở:

```text
D:\Fish_Feeding_Intensity\Dataset\U_FFIA
```

Bên trong cần có 2 thư mục:

```text
D:\Fish_Feeding_Intensity\Dataset\U_FFIA\audio
D:\Fish_Feeding_Intensity\Dataset\U_FFIA\video
```

Cấu trúc file:

```text
audio\date\session\class\*.wav
video\date\session\class\*.mp4
```

Ví dụ:

```text
video\2022_6_13\AM_100\none\13_video_1.mp4
audio\2022_6_13\AM_100\none\13_audio_1.wav
```

Script sẽ tự ghép pair audio-video bằng quy tắc:

```text
13_video_1.mp4 -> 13_audio_1.wav
```

Nếu video không có audio tương ứng, script sẽ báo lỗi.

## 2. Cách chia dữ liệu

Logic lấy từ source gốc U-FFIA:

```text
none   -> label 0
strong -> label 1
medium -> label 2
weak   -> label 3
```

Với mỗi class:

```text
shuffle bằng seed=25
700 mẫu đầu      -> test
700 mẫu tiếp theo -> val
còn lại          -> train
```

Đây là cách chia phân tầng theo class với số mẫu cố định mỗi class, không phải chia theo phần trăm trực tiếp.

Kết quả hiện tại gần tỷ lệ:

```text
train ≈ 79.32%
test  ≈ 10.35%
val   ≈ 10.35%
```

## 3. Cách chạy

Mở terminal tại thư mục project:

```powershell
cd D:\Fish_Feeding_Intensity
```

Chạy:

```powershell
python data_split.py
```

Hoặc chạy bằng đường dẫn đầy đủ:

```powershell
python D:\Fish_Feeding_Intensity\data_split.py
```

## 4. Output sau khi chạy

Script tạo thư mục:

```text
D:\Fish_Feeding_Intensity\Dataset\U_FFIA\splits
```

Bên trong có các file:

```text
train.jsonl
test.jsonl
val.jsonl

train.csv
test.csv
val.csv

summary.json
```

Ý nghĩa:

- `train.jsonl`, `test.jsonl`, `val.jsonl`: mỗi dòng là một sample audio-video pair.
- `train.csv`, `test.csv`, `val.csv`: cùng nội dung, dạng CSV để mở bằng Excel hoặc Pandas.
- `summary.json`: thống kê số lượng mẫu theo split và class.

Ví dụ một sample:

```json
{
  "video_path": "D:\\Fish_Feeding_Intensity\\Dataset\\U_FFIA\\video\\2022_6_13\\AM_100\\none\\13_video_1.mp4",
  "audio_path": "D:\\Fish_Feeding_Intensity\\Dataset\\U_FFIA\\audio\\2022_6_13\\AM_100\\none\\13_audio_1.wav",
  "label": 0,
  "class_name": "none",
  "date": "2022_6_13",
  "session": "AM_100",
  "sample_id": "1"
}
```

## 5. Output thống kê hiện tại

Khi chạy với dataset hiện tại, kết quả là:

```text
Tổng mẫu: 27067

Train: 21467
Test : 2800
Val  : 2800
```

Theo class:

```text
none   : 4871
strong : 7577
medium : 6640
weak   : 7979
```

Split:

```text
Train:
  none   : 3471
  strong : 6177
  medium : 5240
  weak   : 6579

Test:
  none   : 700
  strong : 700
  medium : 700
  weak   : 700

Val:
  none   : 700
  strong : 700
  medium : 700
  weak   : 700
```

## 6. Tuỳ chỉnh tham số

Có thể đổi dataset root:

```powershell
python data_split.py --dataset-root "D:\Fish_Feeding_Intensity\Dataset\U_FFIA"
```

Đổi output folder:

```powershell
python data_split.py --output-dir "D:\Fish_Feeding_Intensity\Dataset\U_FFIA\splits"
```

Đổi seed:

```powershell
python data_split.py --seed 25
```

Đổi số mẫu test/val mỗi class:

```powershell
python data_split.py --test-sample-per-class 700
```

Ví dụ đầy đủ:

```powershell
python data_split.py `
  --dataset-root "D:\Fish_Feeding_Intensity\Dataset\U_FFIA" `
  --output-dir "D:\Fish_Feeding_Intensity\Dataset\U_FFIA\splits" `
  --seed 25 `
  --test-sample-per-class 700
```

## 7. Đọc file split bằng Python

Đọc CSV:

```python
import pandas as pd

train_df = pd.read_csv(r"D:\Fish_Feeding_Intensity\Dataset\U_FFIA\splits\train.csv")
print(train_df.head())
```

Đọc JSONL:

```python
import json

samples = []
with open(r"D:\Fish_Feeding_Intensity\Dataset\U_FFIA\splits\train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

print(samples[0])
```
