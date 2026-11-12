#!/usr/bin/env bash
# Run the offline TensorRT benchmark on the test split of one CV fold.
# The web container is paused during the run so it does not share the GPU,
# and restarted afterwards if it was running.
#
# Usage: FOLD=04 docker/run_benchmark.sh [extra eval_tensorrt.py args]
#   default extra args: --num_samples -1 --workers 1
set -euo pipefail
cd "$(dirname "$0")"

FOLD="${FOLD:-04}"
ENGINE="/app/weights/multimodal_core_fold_${FOLD}_fp32.engine"
MANIFEST="/app/samples/manifests/fold_${FOLD}_test.csv"
OUTPUT="/app/benchmarking/results/fold_${FOLD}"

[ -f "../weights/multimodal_core_fold_${FOLD}_fp32.engine" ] || { echo "Missing engine for fold ${FOLD}"; exit 1; }
[ -f "../samples/manifests/fold_${FOLD}_test.csv" ] || { echo "Missing manifest for fold ${FOLD}"; exit 1; }

WEB_WAS_RUNNING=0
if docker ps -q -f "name=^/aquaffia-edge$" | grep -q .; then
  WEB_WAS_RUNNING=1
  docker compose stop web
fi
restore_web() { [ "${WEB_WAS_RUNNING}" = 1 ] && docker compose start web || true; }
trap restore_web EXIT

if [ "$#" -eq 0 ]; then
  set -- --num_samples -1 --workers 1
fi

mkdir -p "../benchmarking/results/fold_${FOLD}"
docker compose --profile benchmark run --rm benchmark \
  --engine "${ENGINE}" --manifest "${MANIFEST}" --output_dir "${OUTPUT}" "$@"

# The dashboard (/api/benchmark/summary) shows the latest run from results/.
cp -f "../benchmarking/results/fold_${FOLD}"/* ../benchmarking/results/
echo "Results: benchmarking/results/fold_${FOLD}/ (also copied to benchmarking/results/)"
