#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --cov=inference --cov-report=term-missing --cov-fail-under=75
docker compose config --quiet
bash -n scripts/*.sh
# Benchmark evidence: run only when a live inference backend is reachable;
# otherwise the static checks still gate the branch.
BENCH_HEALTH="${INFERENCE_BASE_URL:-http://127.0.0.1:8080}"
BENCH_HEALTH="${BENCH_HEALTH%/v1}"
if curl -fsS -m 3 "${BENCH_HEALTH}/health" >/dev/null 2>&1; then
  INFERENCE_BASE_URL="${INFERENCE_BASE_URL:-http://127.0.0.1:8080/v1}" \
    REQUESTS="${BENCHMARK_REQUESTS:-5}" \
    bash scripts/benchmark.sh
else
  echo "world-class benchmark skipped: no live inference backend at ${BENCH_HEALTH}"
fi
echo "world-class inference checks: PASS"
