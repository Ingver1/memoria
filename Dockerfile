# syntax=docker/dockerfile:1.6
# -------------------------------------------------------------
# Multi‑stage Dockerfile for AI‑memory‑
#   Stage 1: build – install deps with cached pip wheels
#   Stage 2: runtime – copy wheels only → final image < 150 MB
# -------------------------------------------------------------

############################ 1️⃣  build stage ############################
FROM python:3.11-bookworm AS build

# Optional: label for build info
LABEL org.opencontainers.image.source="https://github.com/Ingver1/AI-memory-"

# Install system libs needed for compilation (kept only in build stage)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md /build/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip wheel --no-deps --wheel-dir /wheels .[cli,dev]

# Copy the source after deps to leverage Docker cache
COPY memory_system /build/memory_system
COPY logging.yaml Dockerfile /build/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-deps --wheel-dir /wheels /build

############################ 2️⃣  runtime stage ############################
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_PYTHON_VERSION_WARNING=1

WORKDIR /app

# Install only our own wheels & runtime dependencies
COPY --from=build /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-index --find-links=/wheels ai-memory && \
    pip cache purge

COPY memory_system /app/memory_system
COPY logging.yaml /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD \
  curl -f http://localhost:8000/health/live || exit 1

ENTRYPOINT ["uvicorn", "memory_system.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
