# U_FFIA27K audio

## End-to-end inference benchmark

To benchmark with a saved `audio_best.pt`, set `checkpoint_path`. The benchmark
then uses `checkpoint_path.parent/splits/test.csv` by default. To benchmark
without weights, leave `checkpoint_path` empty and set `test_split_csv_path` to
the test CSV explicitly. In the latter case, predictions use random initialized
weights, but the measured inference path and model architecture are unchanged.

Run from this directory:

```powershell
python benchmark_inference.py
```

If configured, the checkpoint is loaded once and excluded from the measurement.
For each real test WAV, the measured region is disk read, waveform preprocessing,
transfer to the selected device, frontend, and model forward pass. It does not
preload audio into RAM. The first 100 distinct files are warm-up only; the next
1,000 distinct files are timed one at a time (`batch_size = 1`). Results contain
only the mean and sample standard deviation in milliseconds per audio file.
