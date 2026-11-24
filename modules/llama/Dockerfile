FROM python:3.15.0-slim-trixie@sha256:f5b92f2a669a4c347044f29367f31983280b648250ca94dea66f98c53bc87e58

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONOPTIMIZE=2

# libgomp1 and curl are required;
# curl downloads llama-server binary; libgomp1 is the OpenMP runtime
# without it the engine fails at exec with exit 127 during capability probing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Download and install llama-server binary (OpenAI-compatible HTTP server for GGUF models)
# See: https://github.com/ggerganov/llama.cpp/releases
ARG LLAMA_SERVER_VERSION=3.2
RUN curl -L -o /usr/local/bin/llama-server \
    "https://github.com/ggerganov/llama.cpp/releases/download/$LLAMA_SERVER_VERSION/llama-server-linux-x64" \
    && chmod +x /usr/local/bin/llama-server

# Default CMD: run llama-server with model from volume
# Environment variables can override at runtime:
# - LLAMA_MODEL_PATH: path to model directory (default: /models)
# - LLAMA_PORT: server port (default: 8080)
# - LLAMA_HOST: bind host (default: 0.0.0.0)
CMD ["llama-server", "-m", "/models", "-p", "8080", "--host", "0.0.0.0"]

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
  CMD curl -f http://127.0.0.1:8080/health || exit 1