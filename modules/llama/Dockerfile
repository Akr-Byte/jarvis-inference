FROM python:3.15.2-slim-trixie@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONOPTIMIZE=2

# llama.cpp release binary version to install
ARG LLAMA_CPP_RELEASE=b12671
ARG LLAMA_CPP_URL=https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_RELEASE}/llama-${LLAMA_CPP_RELEASE}-bin-ubuntu-x64.tar.gz

# Install runtime deps and download/extract the official prebuilt llama.cpp binary.
# libgomp1 is required by the binary (OpenMP runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fL "${LLAMA_CPP_URL}" -o /tmp/llama.tar.gz \
    && mkdir -p /opt/llama.cpp \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama.cpp --strip-components=1 \
    && rm /tmp/llama.tar.gz \
    && ln -s /opt/llama.cpp/llama-server /usr/local/bin/llama-server \
    && /usr/local/bin/llama-server --version

# Copy the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Run llama-server with model from volume
# Environment variables can be overridden at runtime:
# - LLAMA_MODEL: explicit path to a .gguf file (highest priority)
# - LLAMA_MODEL_PATH: model directory to search for the first .gguf (default: /models)
# - LLAMA_HOST: bind host (default: 0.0.0.0)
# - LLAMA_PORT: server port (default: 8080)
# - LLAMA_ALIAS: model alias exposed by the OpenAI-compatible API
# - LLAMA_EXTRA_ARGS: additional llama-server arguments (passed via env)
ENTRYPOINT ["entrypoint.sh"]
