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

## 2. Web Dashboard (Deployment 1, 2, 3)

* Web Dashboard URL: `http://192.168.55.1:8001` (served by `aquaffia-edge-gateway`)
* Switch deployments with the buttons in the top-right corner, or open `http://192.168.55.1:8001/?d=2`.
* Only ONE deployment runs at a time. Deployment 1 starts by default after boot.
* Architecture and API: `docs/DEPLOYMENT_API.md`. Guide for Deployment 2/3: `deployment_template/README.md`.

| Container | Port | Folder |
|---|---|---|
| `aquaffia-edge-gateway` (page + switching) | 8001 | `deployment_gateway/` |
| `aquaffia-edge` (Deployment 1) | 8011 | `docker/` |
| `aquaffia-edge-2` (Deployment 2) | 8012 | `~/aquaffia_edge_deployment_2` |
| `aquaffia-edge-3` (Deployment 3) | 8013 | `~/aquaffia_edge_deployment_3` |
| `deploy_jetson_web` (other project) | 8000 | `~/Desktop/Deploy_JETSON` |

```bash
# Switch / inspect
curl -X POST http://127.0.0.1:8001/gateway/switch/1      # 1, 2 or 3
curl -s http://127.0.0.1:8001/gateway/state | python3 -m json.tool

# Gateway (once; restarts automatically on boot)
cd deployment_gateway && docker compose up -d --build && cd ..

# Deployment 1 after changing source code (image aquaffia-edge:v1.3.0)
docker/run_web.sh build
docker compose -f docker/docker-compose.yml up --no-start web
curl -X POST http://127.0.0.1:8001/gateway/switch/1
docker/run_web.sh logs
```

Do not start deployment containers with `docker start` / `docker compose up -d`;
let the gateway do it so that only one model is loaded.

---

## 3. Full Test Set Benchmark

Runs the TensorRT pipeline on the test split of one fold (fold 04 by default: 5,412 samples).
Switch the dashboard to Deployment 1 first (so Deployment 2/3 are stopped).
The Deployment 1 container is stopped during the run and restarted afterwards.

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
