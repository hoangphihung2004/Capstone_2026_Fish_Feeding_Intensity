#!/bin/bash
# Shell script to compile ONNX model(s) into TensorRT FP16 engine on NVIDIA Jetson Orin Nano
# Module: model_converter

set -e

WEIGHTS_DIR="/home/fptdanang/fish_feeding/weights"
TIMING_CACHE="${WEIGHTS_DIR}/trt_timing.cache"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
TARGET_FOLD="${1:-all}"

mkdir -p "$WEIGHTS_DIR"

compile_fold() {
    local FOLD_NUM=$1
    local ONNX_FILE="${WEIGHTS_DIR}/multimodal_core_fold_${FOLD_NUM}_sim.onnx"
    local ENGINE_FILE="${WEIGHTS_DIR}/multimodal_core_fold_${FOLD_NUM}_fp16.engine"
    local LOG_FILE="${WEIGHTS_DIR}/build_trt_fold_${FOLD_NUM}.log"

    echo "=========================================================="
    echo ">> [Fold ${FOLD_NUM}] Compiling TensorRT FP16 Engine..."
    echo "   ONNX Input:     $ONNX_FILE"
    echo "   Engine Output:  $ENGINE_FILE"
    echo "   Timing Cache:   $TIMING_CACHE"
    echo "   Log File:       $LOG_FILE"
    echo "=========================================================="

    if [ ! -f "$ONNX_FILE" ]; then
        echo "ERROR: ONNX file not found at $ONNX_FILE"
        exit 1
    fi

    $TRTEXEC \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16 \
        --timingCacheFile="$TIMING_CACHE" \
        --avgRuns=10 \
        --duration=3 \
        2>&1 | tee "$LOG_FILE"

    echo ">> [Fold ${FOLD_NUM}] Completed successfully!"
    ls -lh "$ENGINE_FILE"
    echo ""
}

echo "=========================================================="
echo " Starting TensorRT FP16 Build Process for U_FFIA27K"
echo " Target Fold: $TARGET_FOLD"
echo "=========================================================="

if [ "$TARGET_FOLD" == "all" ]; then
    FOLDS=("00" "01" "02" "03" "04")
    for f in "${FOLDS[@]}"; do
        compile_fold "$f"
    done
    echo "=========================================================="
    echo " All 5 Folds (00 to 04) compiled successfully to TensorRT!"
    echo " Engines in: $WEIGHTS_DIR"
    ls -lh "$WEIGHTS_DIR"/*fp16.engine
    echo "=========================================================="
else
    # Pad fold number to 2 digits if single digit passed
    FOLD_PADDED=$(printf "%02d" "$TARGET_FOLD")
    compile_fold "$FOLD_PADDED"
fi

