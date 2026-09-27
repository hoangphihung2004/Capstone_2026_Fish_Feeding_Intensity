#!/usr/bin/env bash
set -euo pipefail

IMAGE="${AQUAFFIA_IMAGE:-hoanghung442004/aquaffia-edge:v1.1.0}"
CONTAINER="aquaffia-edge-v11"

if docker ps -aq -f "name=^/${CONTAINER}$" | grep -q .; then
  docker start "${CONTAINER}"
  exit 0
fi

mkdir -p /home/fptdanang/cache_v11 /home/fptdanang/results_v11

docker run -d \
  --name "${CONTAINER}" \
  --restart unless-stopped \
  --runtime nvidia \
  --network host \
  -v /home/fptdanang/AquaFFIA_data/samples:/app/samples:ro \
  -v /home/fptdanang/multimodal_core_fold_00_fp32.engine:/app/weights/multimodal_core_fold_00_fp32.engine:ro \
  -v /home/fptdanang/cache_v11:/app/cache \
  -v /home/fptdanang/results_v11:/app/benchmarking/results \
  -v /lib/aarch64-linux-gnu/libnvinfer.so.10:/lib/aarch64-linux-gnu/libnvinfer.so.10:ro \
  -v /lib/aarch64-linux-gnu/libnvinfer_plugin.so.10:/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10:ro \
  -v /lib/aarch64-linux-gnu/libnvonnxparser.so.10:/lib/aarch64-linux-gnu/libnvonnxparser.so.10:ro \
  -v /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudla.so.1:/lib/aarch64-linux-gnu/libcudla.so.1:ro \
  -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/local/lib/python3.10/dist-packages/tensorrt:ro \
  -v /home/fptdanang/.local/lib/python3.10/site-packages/cuda:/usr/local/lib/python3.10/dist-packages/cuda:ro \
  -v /home/fptdanang/.local/lib/python3.10/site-packages/cuda_python-12.9.7.dist-info:/usr/local/lib/python3.10/dist-packages/cuda_python-12.9.7.dist-info:ro \
  -v /home/fptdanang/.local/lib/python3.10/site-packages/cuda_bindings-12.9.9.dist-info:/usr/local/lib/python3.10/dist-packages/cuda_bindings-12.9.9.dist-info:ro \
  "${IMAGE}"
