#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${INFERENCE_BASE_URL:-http://127.0.0.1:8080/v1}"
KEY="${INFERENCE_API_KEY:-}"
REQUESTS="${REQUESTS:-5}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

MODEL="${MODEL:-${DEFAULT_MODEL:-${MODEL_REGISTRY%%,*}}}"
MODEL="${MODEL:-default-model}"

python3 - "$BASE_URL" "$KEY" "$MODEL" "$REQUESTS" <<'PY'
import json
import statistics
import sys
import time
import urllib.request

base, key, model, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
headers = {"Content-Type": "application/json"}
if key:
    headers["Authorization"] = f"Bearer {key}"
body = json.dumps(
    {
        "model": model,
        "messages": [{"role": "user", "content": "Return exactly: benchmark-ok"}],
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
).encode()

totals, ttfts, rates = [], [], []
for _ in range(count):
    req = urllib.request.Request(base + "/chat/completions", body, headers)
    start = time.perf_counter()
    ttft = None
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if ttft is None:
                content = event.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    ttft = time.perf_counter() - start
            usage = event.get("usage")
            if usage:
                completion_tokens = int(usage.get("completion_tokens") or 0)
    total = time.perf_counter() - start
    totals.append(total)
    ttfts.append(ttft if ttft is not None else total)
    decode_seconds = max(0.001, total - ttfts[-1])
    rates.append(completion_tokens / decode_seconds if completion_tokens else 0.0)


def summary(values):
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 3),
        "median": round(statistics.median(ordered), 3),
        "max": round(ordered[-1], 3),
    }


print(
    json.dumps(
        {
            "model": model,
            "requests": count,
            "total_s": summary(totals),
            "ttft_s": summary(ttfts),
            "decode_tokens_per_s": summary(rates),
        },
        indent=2,
    )
)
PY
