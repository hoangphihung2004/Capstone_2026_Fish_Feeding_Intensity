"""
Minimal TensorRT 10 runner for Jetson (cuda-python), usable from model.py.

    from trt_helper import TRTModule
    core = TRTModule("/srv/deployment/weights/my_model_fp32.engine")
    logits = core.infer(audio_features=a, video_form=v)   # by tensor name
    logits = core.infer(a, v)                             # or by input order

Engines must have fixed input shapes (batch 1), like the AquaFFIA engines.
Print core.describe() to see tensor names, shapes and dtypes.
"""

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

_H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
_D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


def _check(result):
    err = result[0] if isinstance(result, tuple) else result
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


class TRTModule:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = _check(cudart.cudaStreamCreate())
        self.inputs, self.outputs = [], []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise ValueError(f"Tensor '{name}' has a dynamic shape {shape}; rebuild the engine with fixed shapes.")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            ptr = _check(cudart.cudaMalloc(nbytes))
            self.context.set_tensor_address(name, int(ptr))
            spec = {"name": name, "shape": shape, "dtype": dtype, "nbytes": nbytes, "ptr": ptr}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(spec)
            else:
                spec["host"] = np.zeros(shape, dtype=dtype)
                self.outputs.append(spec)

    def describe(self) -> str:
        rows = [f"IN  {s['name']} {s['shape']} {s['dtype']}" for s in self.inputs]
        rows += [f"OUT {s['name']} {s['shape']} {s['dtype']}" for s in self.outputs]
        return "\n".join(rows)

    def infer(self, *arrays, **named):
        if arrays and named:
            raise ValueError("Pass inputs either by position or by name, not both.")
        if named:
            arrays = [named[s["name"]] for s in self.inputs]
        if len(arrays) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(arrays)}.")

        for spec, arr in zip(self.inputs, arrays):
            arr = np.ascontiguousarray(arr, dtype=spec["dtype"]).reshape(spec["shape"])
            _check(cudart.cudaMemcpy(spec["ptr"], arr.ctypes.data, spec["nbytes"], _H2D))
        self.context.execute_async_v3(self.stream)
        _check(cudart.cudaStreamSynchronize(self.stream))

        results = []
        for spec in self.outputs:
            _check(cudart.cudaMemcpy(spec["host"].ctypes.data, spec["ptr"], spec["nbytes"], _D2H))
            results.append(spec["host"].copy())
        return results[0] if len(results) == 1 else results
