# AquaFFIA Edge runtime image (Jetson Orin, ARM64, JetPack 6 / L4T R36).
#
# The base image only provides the Python dependencies (torch, torchaudio,
# fastapi, onnxruntime, decord, ...). All application source is copied from
# this repository, so every image matches one commit exactly.
#
# Build from the repository root:
#   docker compose -f docker/docker-compose.yml build web
#
# TensorRT engines, samples, cache and results are NOT baked in; they are
# mounted at runtime by docker/docker-compose.yml.
ARG BASE_IMAGE=hoanghung442004/aquaffia-edge:v1.1.0
FROM ${BASE_IMAGE}

WORKDIR /app

# Remove the stale source, test scripts and artefacts left in the base image.
RUN rm -rf /app/config /app/features /app/models /app/runtime \
           /app/benchmarking /app/web_dashboard /app/samples /app/weights \
           /app/test_*.py /app/export_video_frontend.py /app/video_frontend.onnx

COPY config/ /app/config/
COPY features/ /app/features/
COPY models/ /app/models/
COPY runtime/ /app/runtime/
COPY benchmarking/ /app/benchmarking/
COPY web_dashboard/ /app/web_dashboard/

RUN mkdir -p /app/weights /app/samples /app/cache /app/benchmarking/results

ENV PYTHONUNBUFFERED=1 \
    AQUAFFIA_INPUT_CACHE=1 \
    AQUAFFIA_CACHE_DIR=/app/cache

EXPOSE 8001
CMD ["uvicorn", "web_dashboard.app:app", "--host", "0.0.0.0", "--port", "8001"]
