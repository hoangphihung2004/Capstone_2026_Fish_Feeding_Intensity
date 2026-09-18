# U_FFIA27K audio

## End-to-end inference benchmark

To benchmark with a saved `audio_best.pt`, set `checkpoint_path`. To benchmark
without weights, leave `checkpoint_path` empty. In the latter case, predictions
use random initialized weights, but the measured inference path and model
architecture are unchanged. No sample CSV path is required: the script creates
the test split in memory using the same `dataset_splitter` settings as the audio
pipeline. If cross-validation is configured with `fold_index: null`, it uses
fold `00` deterministically.

Run from this directory:

```powershell
python benchmark_inference.py
```

If configured, the checkpoint is loaded once and excluded from the measurement.
For each real test WAV, the measured region is disk read, waveform preprocessing,
transfer to the selected device, frontend, and model forward pass. It does not
preload audio into RAM or create/delete split CSV files. The first 100 distinct
files are warm-up only; the next 1,000 distinct files are timed one at a time
(`batch_size = 1`). Results contain only the mean and sample standard deviation
in milliseconds per audio file.
