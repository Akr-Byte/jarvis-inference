#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/docker-compose.yml"
ENV_FILE="$ROOT/.env"

cd "$ROOT"

fail() {
  echo "ERROR: $1" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || fail "Docker is required."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."
command -v curl >/dev/null 2>&1 || fail "curl is required."

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/.env.example" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "Created .env from .env.example."
  echo "Set INFERENCE_API_KEY before allowing remote access."
fi

set -a
# shellcheck disable=SC1091
source "$ENV_FILE"
set +a

bind_address="${INFERENCE_BIND_ADDRESS:-127.0.0.1}"
port="${INFERENCE_PORT:-8080}"
api_key="${INFERENCE_API_KEY:-}"
backend="${INFERENCE_BACKEND:-llamacpp}"

case "${backend,,}" in
  llamacpp|llama.cpp)
    ;;
  ollama)
    fail "This installer targets the default llama.cpp stack. For Ollama, use: docker compose -f docker-compose.ollama.yml up -d ollama"
    ;;
  *)
    fail "Unknown INFERENCE_BACKEND: $backend"
    ;;
esac

if [[ "$bind_address" != "127.0.0.1" && "$bind_address" != "::1" && -z "$api_key" ]]; then
  fail "INFERENCE_API_KEY is required when the gateway is remotely reachable."
fi

shopt -s nullglob
gguf_models=("$ROOT"/models/*.gguf)
shopt -u nullglob
(( ${#gguf_models[@]} > 0 )) || {
  fail "No GGUF model found in ./models. Copy at least one .gguf file there before install."
}
echo "==> Found ${#gguf_models[@]} GGUF model(s) in ./models"

echo "==> Cleaning stale inference containers"
# Compose uses stable container names. Remove only this application containers;
# the model directory is preserved.
for container in jarvis-inference jarvis-llama-server; do
  if docker container inspect "$container" >/dev/null 2>&1; then
    echo "Removing existing container: $container"
    docker rm -f "$container" >/dev/null
  fi
done

echo "==> Validating Compose configuration"
docker compose -f "$COMPOSE_FILE" config --quiet

echo "==> Building llama-server and inference gateway"
docker compose -f "$COMPOSE_FILE" build --pull llama-server jarvis-inference

echo "==> Starting inference services"
docker compose -f "$COMPOSE_FILE" up -d llama-server jarvis-inference

echo "==> Waiting for llama-server"
llama_ready=0
for _ in {1..60}; do
  if curl --fail --silent --show-error "http://127.0.0.1:8081/health" >/dev/null 2>&1; then
    llama_ready=1
    break
  fi
  sleep 2
done
(( llama_ready == 1 )) || {
  docker compose -f "$COMPOSE_FILE" ps
  docker compose -f "$COMPOSE_FILE" logs llama-server
  fail "llama-server did not become ready. Check the mounted GGUF model and logs."
}

echo "==> Waiting for inference readiness"
ready=0
for _ in {1..60}; do
  if curl --fail --silent --show-error "http://127.0.0.1:$port/ready" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
(( ready == 1 )) || {
  docker compose -f "$COMPOSE_FILE" ps
  docker compose -f "$COMPOSE_FILE" logs jarvis-inference
  fail "Inference gateway did not become ready."
}

echo "==> Running smoke checks"
curl --fail --silent --show-error "http://127.0.0.1:$port/health" >/dev/null
if [[ -n "$api_key" ]]; then
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $api_key" \
    "http://127.0.0.1:$port/v1/models" >/dev/null
else
  curl --fail --silent --show-error \
    "http://127.0.0.1:$port/v1/models" >/dev/null
fi

echo
docker compose -f "$COMPOSE_FILE" ps
echo
echo "Installation complete."
echo "Gateway: http://127.0.0.1:$port"
