FROM python:3.15.0-slim-trixie@sha256:f5b92f2a669a4c347044f29367f31983280b648250ca94dea66f98c53bc87e58

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONOPTIMIZE=2

# libgomp1 is required by the managed llama-server binary (OpenMP runtime);
# without it the engine fails at exec with exit 127 during capability probing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# llama-server binary is expected at /usr/local/bin/llama-server
# via volume mount from host: ./llama-server:/usr/local/bin/llama-server
# If the binary is not present, the container will keep running so the
# volume can be attached at runtime.

# Run llama-server if binary exists; otherwise keep container alive
# so the volume can be attached at runtime.
CMD ["/bin/sh", "-c", "if [ -x /usr/local/bin/llama-server ]; then \
    exec llama-server -m /models -p 8080 --host 0.0.0.0; \
    else echo 'llama-server binary not found - mount ./llama-server:/usr/local/bin/llama-server'; \
    tail -f /dev/null; fi"]