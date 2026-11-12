# AquaFFIA Edge - Operational Commands

Quick reference for the web dashboard and benchmark on NVIDIA Jetson Orin Nano.
All commands run from the project folder on the Jetson:

```bash
cd ~/aquaffia_edge_deployment
```

---

## 0. Layout on the Jetson

```
aquaffia_edge_deployment/          source code (this repository)
├── weights/                       TensorRT engines: multimodal_core_fold_0X_fp32.engine,
│                                  audio_frontend_fp32.engine, video_frontend_fp32.engine
├── samples/
│   ├── manifest.csv               default manifest (fold 04 test split)
│   ├── manifests/fold_0X_test.csv test split of each CV fold
│   ├── continuous_simulation/     598 s stream for the real-time dashboard
│   ├── video/  audio/             mount points; the dataset is mounted here at runtime
├── cache/                         input tensor cache (safe to delete)
└── benchmarking/results/fold_0X/  benchmark outputs
```

The dataset is read from `~/Desktop/Fish_Feeding_Intensity_Dataset`
(override with `DATASET_DIR=...`).

---

## 1. First-time data setup

```bash
python3 data_preparation/build_manifests.py \
    --dataset_dir ~/Desktop/Fish_Feeding_Intensity_Dataset --output_dir samples --default_fold 04
bash data_preparation/build_continuous_simulation.sh \
    ~/Desktop/Fish_Feeding_Intensity_Dataset samples
```

---

## 2. Web Dashboard

```bash
docker/run_web.sh build      # build image aquaffia-edge:v1.2.0 from source
docker/run_web.sh up         # start (restarts automatically on boot)
docker/run_web.sh stop
docker/run_web.sh restart
docker/run_web.sh logs
docker/run_web.sh status
```

* Web Dashboard URL: `http://192.168.55.1:8001`
  (port 8000 is used by `deploy_jetson_web`; change with `AQUAFFIA_PORT=...`)

After changing source code: `docker/run_web.sh build && docker/run_web.sh up`.

---

## 3. Full Test Set Benchmark

Runs the TensorRT pipeline on the test split of one fold (fold 04 by default: 5,412 samples).
The web container is stopped during the run and restarted afterwards.

```bash
docker/run_benchmark.sh                          # fold 04, full test set
FOLD=00 docker/run_benchmark.sh                  # another fold
docker/run_benchmark.sh --num_samples 200 --stratified --workers 1   # quick check
```

* Results: `benchmarking/results/fold_0X/benchmark_results.json`
* Confusion matrix: `benchmarking/results/fold_0X/confusion_matrix.png`
* The latest run is also copied to `benchmarking/results/`, which the dashboard displays.

---

## 4. Maximum Performance Mode (Lock Clocks)

Lock GPU (765 MHz) and CPU (1.98 GHz) to maximum frequencies to prevent dynamic power-saving throttling (DVFS) and achieve optimal latency (reduces model inference from ~25 ms down to ~11–12 ms):

```bash
sudo jetson_clocks
```

Inspect current clock frequencies and governor status:

```bash
sudo jetson_clocks --show
```
