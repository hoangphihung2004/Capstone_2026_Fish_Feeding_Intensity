import os
import sys
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

engine_path = "/app/weights/multimodal_core_fold_00_fp32.engine"
if not os.path.exists(engine_path):
    engine_path = "/home/fptdanang/multimodal_core_fold_00_fp32.engine"
if not os.path.exists(engine_path):
    print(f"Error: {engine_path} does not exist!")
    sys.exit(1)

logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

print("=== Engine Info ===")
print("num_io_tensors:", engine.num_io_tensors)
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    print(f"  [{i}] {name}: mode={mode}, shape={shape}, dtype={dtype}")

context = engine.create_execution_context()

# Allocate CUDA buffers
# audio_features: [1, 1, 128, 128] float32 = 16384 * 4 = 65536 bytes
# video_form: [1, 6, 224, 224] float32 = 301056 * 4 = 1204224 bytes
# clipwise_output: [1, 4] float32 = 4 * 4 = 16 bytes

res_audio = cudart.cudaMalloc(65536)
print("cudaMalloc audio result:", res_audio)
res_video = cudart.cudaMalloc(1204224)
print("cudaMalloc video result:", res_video)
res_out = cudart.cudaMalloc(16)
print("cudaMalloc out result:", res_out)
err_stream, stream = cudart.cudaStreamCreate()
print("cudaStreamCreate result:", err_stream, stream)

d_audio = res_audio[1]
d_video = res_video[1]
d_out = res_out[1]

context.set_tensor_address("audio_features", int(d_audio))
context.set_tensor_address("video_form", int(d_video))
context.set_tensor_address("clipwise_output", int(d_out))

# Dummy inputs
dummy_audio = np.random.randn(1, 1, 128, 128).astype(np.float32)
dummy_video = np.random.randn(1, 6, 224, 224).astype(np.float32)
h_out = np.zeros((1, 4), dtype=np.float32)

# Copy host to device
cudart.cudaMemcpy(d_audio, dummy_audio.ctypes.data, 65536, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
cudart.cudaMemcpy(d_video, dummy_video.ctypes.data, 1204224, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

# Warmup and benchmark
import time
for _ in range(5):
    context.execute_async_v3(stream)
cudart.cudaStreamSynchronize(stream)

times = []
for _ in range(50):
    t0 = time.perf_counter()
    context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    times.append((time.perf_counter() - t0) * 1000)

cudart.cudaMemcpy(h_out.ctypes.data, d_out, 16, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

print(f"TRT 10 FP32 Inference successful!")
print(f"Mean GPU time over 50 runs: {np.mean(times):.2f} ms ({1000.0/np.mean(times):.1f} FPS)")
print(f"Output shape: {h_out.shape}, Output logits: {h_out}")
