# ==============================================================================
# AquaFFIA Edge Deployment - NVIDIA Jetson Orin Nano (MAXN Mode)
# Base Image: NVIDIA JetPack 6.x (L4T r36.2.0, Ubuntu 22.04 LTS, aarch64)
# ==============================================================================
FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system utilities and multimedia codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 1b. Install NVIDIA TensorRT & CUDA runtime libraries from JetPack 6 L4T repo
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/jetson-ota-public.gpg && \
    echo "deb https://repo.download.nvidia.com/jetson/common r36.5 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    echo "deb https://repo.download.nvidia.com/jetson/t234 r36.5 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvonnxparsers10 \
    python3-libnvinfer \
    libcudla-12-6 \
    && echo "/usr/local/cuda-12.6/targets/aarch64-linux/lib" > /etc/ld.so.conf.d/cuda.conf \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and build tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir "cuda-python<13.0"

# 3. Install core web, audio, and computer vision libraries
RUN pip3 install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.29.0 \
    soundfile==0.12.1 \
    opencv-python-headless \
    pandas==2.2.1 \
    jinja2 \
    python-multipart \
    onnx \
    onnxruntime==1.17.3

# 4. CUDA-enabled PyTorch for JetPack 6 / CUDA 12.6 on Jetson.
# The NVIDIA Jetson AI Lab index serves aarch64 CUDA wheels; do not replace
# this with PyTorch's CPU-only index or AudioFrontend will run on the CPU.
ARG JETSON_TORCH_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
ARG JETSON_TORCH_VERSION=2.8.0
RUN pip3 install --no-cache-dir --index-url ${JETSON_TORCH_INDEX} \
    torch==${JETSON_TORCH_VERSION} torchaudio==${JETSON_TORCH_VERSION}

# 5. Install audio frontend signal processing dependencies
RUN pip3 install --no-cache-dir torchlibrosa

# 6. Bake complete application source code, models, and configurations into container
COPY config/ /app/config/
COPY features/ /app/features/
COPY models/ /app/models/
COPY runtime/ /app/runtime/
COPY weights/ /app/weights/
COPY checkpoint/multimodal_model/fold_00/multimodal_best.pt /app/checkpoint/multimodal_model/fold_00/multimodal_best.pt
COPY web_dashboard/ /app/web_dashboard/
COPY benchmarking/ /app/benchmarking/
RUN mkdir -p /app/samples
RUN mkdir -p /app/cache /app/benchmarking/results
COPY samples/manifest.csv /app/samples/manifest.csv

EXPOSE 8000

CMD ["uvicorn", "web_dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
