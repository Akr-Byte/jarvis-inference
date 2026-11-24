#!/bin/sh
set -e

# Resolve the GGUF model path.
# Priority: LLAMA_MODEL (explicit file) > first *.gguf found under LLAMA_MODEL_PATH.
MODEL_PATH="${LLAMA_MODEL:-${LLAMA_MODEL_PATH:-/models}}"
notified=0

while true; do
    if [ -d "$MODEL_PATH" ]; then
        MODEL_FILE=$(find "$MODEL_PATH" -maxdepth 2 -name '*.gguf' -type f 2>/dev/null | head -n 1)
        if [ -n "$MODEL_FILE" ]; then
            MODEL_PATH="$MODEL_FILE"
            break
        fi
        if [ "$notified" -eq 0 ]; then
            echo "No GGUF model found in $MODEL_PATH"
            echo "Mount a model at ./models:/models:ro, for example: cp /path/to/model.gguf ./models/"
            echo "Waiting for a model to appear..."
            notified=1
        fi
    elif [ -f "$MODEL_PATH" ]; then
        break
    elif [ "$notified" -eq 0 ]; then
        echo "GGUF model path does not exist: $MODEL_PATH"
        echo "Mount a model at ./models:/models:ro or set LLAMA_MODEL to a valid file."
        echo "Waiting for a model to appear..."
        notified=1
    fi
    sleep 10
done

echo "Starting llama-server with model: $MODEL_PATH on ${LLAMA_HOST:-0.0.0.0}:${LLAMA_PORT:-8080}"
ALIAS_ARGS=""
if [ -n "${LLAMA_ALIAS:-}" ]; then
    ALIAS_ARGS="--alias ${LLAMA_ALIAS}"
fi
# shellcheck disable=SC2086
exec llama-server -m "$MODEL_PATH" \
    --host "${LLAMA_HOST:-0.0.0.0}" \
    --port "${LLAMA_PORT:-8080}" \
    ${ALIAS_ARGS} \
    ${LLAMA_EXTRA_ARGS:-}
