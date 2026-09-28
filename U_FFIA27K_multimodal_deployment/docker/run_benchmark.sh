#!/usr/bin/env bash
set -euo pipefail

IMAGE="${AQUAFFIA_IMAGE:-hoanghung442004/aquaffia-edge:v1.1.0}"
WEB_CONTAINER="aquaffia-edge"

if docker ps -q -f "name=^/${WEB_CONTAINER}$" | grep -q .; then
  docker stop "${WEB_CONTAINER}"
fi

RESULTS_DIR="${AQUAFFIA_RESULTS_DIR:-/home/fptdanang/aquaffia-edge/results}"
mkdir -p "${RESULTS_DIR}"

if [ "$#" -eq 0 ]; then
  set -- --num_samples -1 --workers 1
fi

docker run --rm \
  --runtime nvidia \
  --network host \
  -e PYTHONUNBUFFERED=1 \
  -v /home/fptdanang/AquaFFIA_data/samples:/app/samples:ro \
  -v /home/fptdanang/multimodal_core_fold_00_fp32.engine:/app/weights/multimodal_core_fold_00_fp32.engine:ro \
  -v /home/fptdanang/audio_frontend_fp32.engine:/app/weights/audio_frontend_fp32.engine:ro \
  -v /home/fptdanang/video_frontend_fp32.engine:/app/weights/video_frontend_fp32.engine:ro \
  -v "${RESULTS_DIR}:/app/benchmarking/results" \
  -v /lib/aarch64-linux-gnu/libnvinfer.so.10:/lib/aarch64-linux-gnu/libnvinfer.so.10:ro \
  -v /lib/aarch64-linux-gnu/libnvinfer_plugin.so.10:/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10:ro \
  -v /lib/aarch64-linux-gnu/libnvonnxparser.so.10:/lib/aarch64-linux-gnu/libnvonnxparser.so.10:ro \
  -v /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudla.so.1:/lib/aarch64-linux-gnu/libcudla.so.1:ro \
  -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/local/lib/python3.10/dist-packages/tensorrt:ro \
  -v /home/fptdanang/.local/lib/python3.10/site-packages/cuda:/usr/local/lib/python3.10/dist-packages/cuda:ro \
  -v /home/fptdanang/.local/lib/python3.10/site-packages/cuda_python-12.9.7.dist-info:/usr/local/lib/python3.10/dist-packages/cuda_python-12.9.7.dist-info:ro \
  -v /home/fptdanang/.local/lib/python3.10/site-packages/cuda_bindings-12.9.9.dist-info:/usr/local/lib/python3.10/dist-packages/cuda_bindings-12.9.9.dist-info:ro \
  "${IMAGE}" \
  python3 /app/benchmarking/eval_tensorrt.py "$@"
