# ARM64 release image, built on the Jetson from the previously verified image.
# It replaces runtime source without copying test media.
FROM hoanghung442004/aquaffia-edge:latest

COPY runtime/ /app/runtime/
COPY benchmarking/eval_tensorrt.py /app/benchmarking/eval_tensorrt.py
COPY web_dashboard/app.py /app/web_dashboard/app.py

RUN mkdir -p /app/cache /app/benchmarking/results

ENV AQUAFFIA_INPUT_CACHE=1 \
    AQUAFFIA_CACHE_DIR=/app/cache
