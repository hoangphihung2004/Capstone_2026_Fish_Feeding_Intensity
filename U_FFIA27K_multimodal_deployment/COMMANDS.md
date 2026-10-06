# AquaFFIA Edge - Operational Commands

Quick reference commands for managing the web container and running the full benchmark evaluation on NVIDIA Jetson Orin Nano.

---

## 1. Start Web Dashboard Container

Start the real-time simulation and web dashboard service:

```bash
docker start aquaffia-edge
```

* Web Dashboard URL: `http://192.168.55.1:8000`

---

## 2. Stop Web Dashboard Container

Stop the running web dashboard service:

```bash
docker stop aquaffia-edge
```

---

## 3. View Container Logs

Monitor real-time logs from the web container:

```bash
docker logs -f aquaffia-edge
```

To inspect the last 50 lines without continuous streaming:

```bash
docker logs --tail 50 aquaffia-edge
```

---

## 4. Run Full Test Set Benchmark

Execute the offline benchmark evaluation across the entire test set (5,415 samples, Fold 00):

```bash
/home/fptdanang/run_benchmark.sh --num_samples -1 --workers 1
```

* Results and metrics will be saved to: `/home/fptdanang/aquaffia-edge/results/benchmark_results.json`
* Confusion matrix plot will be saved to: `/home/fptdanang/aquaffia-edge/results/confusion_matrix.png`

---

## 5. Maximum Performance Mode (Lock Clocks)

Lock GPU (765 MHz) and CPU (1.98 GHz) to maximum frequencies to prevent dynamic power-saving throttling (DVFS) and achieve optimal latency (reduces model inference from ~25 ms down to ~11–12 ms):

```bash
sudo jetson_clocks
```

Inspect current clock frequencies and governor status:

```bash
sudo jetson_clocks --show
```

