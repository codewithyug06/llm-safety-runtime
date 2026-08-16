# ── ARGUS: LLM Safety Runtime Dockerfile ─────────────────────────────────────
# Multi-stage production container for ARGUS Safety Gateway and Monitoring Engine

FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid heavy CUDA wheels in CPU containers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install core runtime dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    structlog \
    redis \
    aioredis \
    confluent-kafka \
    prometheus-client \
    httpx \
    pyyaml \
    omegaconf \
    scikit-learn \
    scipy \
    mlflow \
    langgraph \
    langchain \
    transformers \
    accelerate \
    opentelemetry-sdk \
    tenacity \
    python-dotenv \
    rich \
    typer \
    tqdm

# ── Final Runtime Stage ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create unprivileged user for security
RUN useradd -m -u 1000 argus && chown -R argus:argus /app

# Copy application codebase
COPY --chown=argus:argus src/ /app/src/
COPY --chown=argus:argus configs/ /app/configs/
COPY --chown=argus:argus pyproject.toml /app/pyproject.toml

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080

USER argus

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:8080/health | grep -q healthy || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
