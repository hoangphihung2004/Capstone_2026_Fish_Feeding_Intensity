# AquaFFIA deployment image (Deployment 2 / 3) for Jetson Orin, ARM64, JetPack 6.
# The base image already contains Python 3.10, torch, torchaudio, OpenCV,
# soundfile, FastAPI, uvicorn and onnxruntime. TensorRT and cuda-python are
# mounted from the Jetson host by docker-compose.yml.
ARG BASE_IMAGE=hoanghung442004/aquaffia-edge:v1.1.0
FROM ${BASE_IMAGE}

WORKDIR /srv/deployment

# Optional extra Python packages: list them in requirements.txt (needs internet).
COPY requirements.txt /srv/deployment/requirements.txt
RUN if [ -s requirements.txt ]; then pip3 install --no-cache-dir -r requirements.txt; fi

COPY . /srv/deployment

ENV PYTHONUNBUFFERED=1 \
    SIM_DIR=/data/continuous_simulation \
    WEIGHTS_DIR=/srv/deployment/weights

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8012}"]
