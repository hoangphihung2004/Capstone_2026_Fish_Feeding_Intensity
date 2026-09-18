# U_FFIA27K audio

## End-to-end inference benchmark

Edit `config/inference_benchmark_config.json` and set `checkpoint_path` to the
saved `audio_best.pt`. The benchmark uses `checkpoint_path.parent/splits/test.csv`
by default; set `test_split_csv_path` only when that CSV is elsewhere.

Run from this directory:

```powershell
python benchmark_inference.py
```

The checkpoint is loaded once and excluded from the measurement. For each real
test WAV, the measured region is disk read, waveform preprocessing, transfer to
the selected device, frontend, and model forward pass. It does not preload audio
into RAM. The first 100 distinct files are warm-up only; the next 1,000 distinct
files are timed one at a time (`batch_size = 1`). Results contain only the mean
and sample standard deviation in milliseconds per audio file.
