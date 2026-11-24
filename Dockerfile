FROM python:3.15.2-slim-trixie@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONOPTIMIZE=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/ ./inference/

# Run the gateway as an unprivileged user (Trivy DS-0002, defense in depth).
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8080

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8080"]
