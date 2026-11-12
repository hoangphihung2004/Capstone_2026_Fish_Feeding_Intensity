# AquaFFIA dashboard gateway: serves the dashboard and switches deployments.
# Reuses the dependency base already on the Jetson (FastAPI + uvicorn); no model is loaded.
ARG BASE_IMAGE=hoanghung442004/aquaffia-edge:v1.1.0
FROM ${BASE_IMAGE}

WORKDIR /gateway
COPY gateway.py /gateway/gateway.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8001
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8001"]
