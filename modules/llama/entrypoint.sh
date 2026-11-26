#!/bin/sh
set -e

# Resolve the GGUF model path.
# Priority: LLAMA_MODEL (explicit file) > first *.gguf found under LLAMA_MODEL_PATH.
MODEL_PATH="${LLAMA_MODEL:-${LLAMA_MODEL_PATH:-/models}}"
notified=0

while true; do
    if [ -d "$MODEL_PATH" ]; then
        models=$(find "$MODEL_PATH" -maxdepth 2 -name '*.gguf' -type f 2>/dev/null | LC_ALL=C sort)
        model_count=$(printf '%s\n' "$models" | grep -c . || true)
        if [ "$model_count" -gt 1 ] && [ "$notified" -eq 0 ]; then
            echo "warning: $model_count GGUF files found; selecting the first in sorted order" >&2
            echo "warning: set LLAMA_MODEL to pin a specific file" >&2
        fi
        MODEL_FILE=$(printf '%s\n' "$models" | head -n 1)
        if [ -n "$MODEL_FILE" ]; then
            MODEL_PATH="$MODEL_FILE"
            break
        fi
        if [ "$notified" -eq 0 ]; then
            echo "No GGUF model found in $MODEL_PATH"
            echo "Mount a GGUF at /models (set MODEL_DIRECTORY on the host) or set LLAMA_MODEL."
            echo "Waiting for a model to appear..."
            notified=1
        fi
    elif [ -f "$MODEL_PATH" ]; then
        break
    elif [ "$notified" -eq 0 ]; then
        echo "GGUF model path does not exist: $MODEL_PATH"
        echo "Mount a GGUF at /models (set MODEL_DIRECTORY on the host) or set LLAMA_MODEL."
        echo "Waiting for a model to appear..."
        notified=1
    fi
    sleep 10
done

# Build the llama-server command. Every optional flag is capability-probed so
# older/newer llama.cpp releases never fail on a renamed option.
HELP_TEXT=$(llama-server --help 2>&1 || true)
ARGS=""

has_flag() {
    # Match the flag as a whole option token, not as a substring of a longer
    # option or a deprecation note in the help text.
    printf '%s\n' "$HELP_TEXT" | grep -Eq -- "(^|[[:space:],])$1([[:space:],=]|$)"
}
skip_flag() {
    echo "warning: llama-server does not support $1; skipping" >&2
}
quote_arg() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}
append() {
    ARGS="$ARGS $(quote_arg "$1")"
}
add_switch() {
    flag=$1
    if has_flag "$flag"; then
        append "$flag"
    else
        skip_flag "$flag"
    fi
}
add_valued() {
    flag=$1
    value=$2
    [ -n "$value" ] || return 0
    if has_flag "$flag"; then
        append "$flag"
        append "$value"
    else
        skip_flag "$flag"
    fi
}
add_first_supported() {
    value=$1
    shift
    [ -n "$value" ] || return 0
    first=$1
    while [ "$#" -gt 0 ]; do
        flag=$1
        shift
        if has_flag "$flag"; then
            append "$flag"
            append "$value"
            return 0
        fi
    done
    skip_flag "$first"
}
is_true() {
    case $(printf '%s' "$1" | tr '[:upper:]' '[:lower:]') in
        1 | true | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}
is_quantized() {
    case $(printf '%s' "$1" | tr '[:upper:]' '[:lower:]') in
        '' | f16 | f32 | bf16) return 1 ;;
        *) return 0 ;;
    esac
}

append -m
append "$MODEL_PATH"
append --host
append "${LLAMA_HOST:-0.0.0.0}"
append --port
append "${LLAMA_PORT:-8080}"

alias="${LLAMA_ALIAS:-${LLAMACPP_ALIAS:-}}"
[ -n "$alias" ] && add_valued --alias "$alias"

context="${LLAMA_CTX_SIZE:-${MAX_CONTEXT_LENGTH:-16384}}"
parallel="${LLAMACPP_PARALLEL:-1}"
case "$context" in '' | *[!0-9]*) context=16384 ;; esac
case "$parallel" in '' | *[!0-9]*) parallel=1 ;; esac
add_valued --ctx-size "$((context * parallel))"

add_valued --threads "${LLAMACPP_THREADS:-}"
add_valued --threads-batch "${LLAMACPP_BATCH_THREADS:-}"
add_valued --batch-size "${LLAMACPP_BATCH_SIZE:-}"
add_valued --ubatch-size "${LLAMACPP_UBATCH_SIZE:-}"
add_valued --parallel "$parallel"

gpu_layers="${LLAMACPP_GPU_LAYERS:-0}"
case "$gpu_layers" in '' | *[!0-9]*) gpu_layers=0 ;; esac
[ "$gpu_layers" -gt 0 ] && add_valued --n-gpu-layers "$gpu_layers"

if is_true "${LLAMACPP_CACHE_PROMPT:-true}"; then
    add_switch --cache-prompt
fi
cache_ram="${LLAMACPP_CACHE_RAM_MIB:-0}"
case "$cache_ram" in '' | *[!0-9]*) cache_ram=0 ;; esac
# llama-server's --cache-ram unit is MiB; 0 explicitly disables the cache
# instead of leaving the engine's 8192 MiB default in place.
add_valued --cache-ram "$cache_ram"
cache_reuse="${LLAMACPP_CACHE_REUSE:-0}"
case "$cache_reuse" in '' | *[!0-9]*) cache_reuse=0 ;; esac
[ "$cache_reuse" -gt 0 ] && add_valued --cache-reuse "$cache_reuse"
cache_k="${LLAMACPP_CACHE_TYPE_K:-}"
cache_v="${LLAMACPP_CACHE_TYPE_V:-}"
add_valued --cache-type-k "$cache_k"
add_valued --cache-type-v "$cache_v"

if is_true "${LLAMACPP_FLASH_ATTENTION:-false}" || is_quantized "$cache_k" || is_quantized "$cache_v"; then
    add_valued --flash-attn auto
fi

if is_true "${LLAMACPP_JINJA:-true}"; then
    add_switch --jinja
else
    add_switch --no-jinja
fi
if is_true "${LLAMACPP_METRICS:-true}"; then
    add_switch --metrics
fi
if is_true "${LLAMACPP_MLOCK:-false}"; then
    add_valued --load-mode mlock
fi

reasoning="${LLAMACPP_REASONING_FORMAT:-}"
case "$reasoning" in
    auto | none | deepseek) add_valued --reasoning-format "$reasoning" ;;
    '') ;;
    *) echo "warning: ignoring unsupported LLAMACPP_REASONING_FORMAT=$reasoning" >&2 ;;
esac

draft_model="${LLAMACPP_DRAFT_MODEL:-}"
if [ -n "$draft_model" ]; then
    add_first_supported "$draft_model" --spec-draft-model --draft-model
    draft_max="${LLAMACPP_DRAFT_MAX_LEN:-5}"
    case "$draft_max" in '' | *[!0-9]*) draft_max=5 ;; esac
    draft_min="${LLAMACPP_DRAFT_MIN_LEN:-5}"
    case "$draft_min" in '' | *[!0-9]*) draft_min=5 ;; esac
    [ "$draft_min" -gt "$draft_max" ] && draft_min=$draft_max
    add_first_supported "$draft_max" --spec-draft-n-max --draft-max-len
    add_first_supported "$draft_min" --spec-draft-n-min --draft-min-len
    add_valued --draft-ctx-size "${LLAMACPP_DRAFT_CTX_SIZE:-}"
fi

eval "set -- $ARGS"
# LLAMA_EXTRA_ARGS is intentionally word-split for arbitrary engine arguments.
# shellcheck disable=SC2086
set -- "$@" ${LLAMA_EXTRA_ARGS:-}

echo "Starting llama-server with model: $MODEL_PATH on ${LLAMA_HOST:-0.0.0.0}:${LLAMA_PORT:-8080}"
exec llama-server "$@"
