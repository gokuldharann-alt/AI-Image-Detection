FROM python:3.11-slim

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (Pillow/jpeg runtime + curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch (~200MB) instead of default CUDA build (~2.5GB).
# Install torch first from CPU index, then the rest of requirements.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    grep -v -E "^(torch|torchvision)" requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt

COPY . .

# Render injects $PORT dynamically — do NOT hardcode. 224px + torch path
# keeps RSS under the 512MB free-tier limit (384px + ONNX overflows it).
ENV DEVICE=cpu \
    USE_ONNX=0 \
    USE_FP16=0 \
    IMAGE_SIZE=224 \
    HOST=0.0.0.0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
