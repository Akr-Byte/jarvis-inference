FROM python:3.15.2-slim-trixie@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONOPTIMIZE=2

# Ollama requires these runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Ollama will be installed and configured at runtime
# The entrypoint sets up the ollama service
CMD ["ollama", "serve"]

EXPOSE 11434

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -f http://127.0.0.1:11434/api/tags || exit 1