#!/bin/bash
# Shell script to compile ONNX model into TensorRT FP16 engine on Jetson Orin Nano
# Module: deployment.model_converter

set -e

ONNX_PATH="${1:-/home/fptdanang/fish_feeding/weights/multimodal_core_sim.onnx}"
ENGINE_PATH="${2:-/home/fptdanang/fish_feeding/weights/multimodal_core_fp16.engine}"
LOG_PATH="/home/fptdanang/fish_feeding/weights/build_trt.log"

echo "=========================================================="
echo "Compiling TensorRT FP16 Engine for Jetson Orin Nano"
echo "Input ONNX:    $ONNX_PATH"
echo "Output Engine: $ENGINE_PATH"
echo "Log file:      $LOG_PATH"
echo "=========================================================="

mkdir -p "$(dirname "$ENGINE_PATH")"

/usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --fp16 \
    --avgRuns=10 \
    --duration=3 \
    2>&1 | tee "$LOG_PATH"

echo "TensorRT Engine compilation finished successfully!"
ls -lh "$ENGINE_PATH"
