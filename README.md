# jarvis-inference

**Production-grade local inference gateway for the Jarvis stack.**

[![CI](https://github.com/abdullahalrifat/jarvis-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullahalrifat/jarvis-inference/actions/workflows/ci.yml)
[![Release](https://github.com/abdullahalrifat/jarvis-inference/releases/badge.svg)](https://github.com/abdullahalrifat/jarvis-inference/releases)
[![PyPI](https://img.shields.io/pypi/v/jarvis-inference)](https://pypi.org/project/jarvis-inference/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-beta-blue)](https://github.com/abdullahalrifat/jarvis-inference/releases)

The dedicated inference VM owns model execution for the Jarvis stack. This repository is the
OpenAI-compatible HTTP gateway: an agent-facing API that schedules bounded generations against a
local model backend (Ollama, or an optional llama.cpp server) and speaks the OpenAI wire format.

- Agent consumers — the standalone **Jarvis CLI** and the optional **AI Stack** control plane — are
  independent sibling clients of this gateway. Neither runs the model backend itself.
- The gateway owns the model allowlist, bounded queue, concurrency limit, backend timeouts and
  resource limits.
- Jarvis remains usable for local repository work when AI Stack is stopped, and AI Stack remains
  usable when the CLI is not installed.

**Current release: `0.5.1`** — see [CHANGELOG.md](CHANGELOG.md) for release notes.

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Install](#install)
- [Configuration](#configuration)
- [Production and constrained-host profiles](#production-and-constrained-host-profiles)
- [Queueing and CPU-only operation](#queueing-and-cpu-only-operation)
- [Embedding efficiency](#embedding-efficiency)
- [Reasoning models](#reasoning-models)
- [Deployment lifecycle](#deployment-lifecycle)
- [Operations](#operations)
- [Model integrity](#model-integrity)
- [Security](#security)
- [Development](#development)
- [Operational validation](#operational-validation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

## Overview

`jarvis-inference` is a production-grade, CPU-only inference gateway for small local language
models. It exposes an OpenAI-compatible API:

- `POST /v1/chat/completions` — chat completions with optional streaming (SSE), tool calls and
  reasoning content.
- `POST /v1/embeddings` — embedding generation with batched backend submission.
- `GET /v1/models` — model listing for the configured registry.
- `GET /v1/capabilities` — capability advertisement for clients.
- `GET /v1/inference/status` — in-process inference/runtime status.
- `GET /health` and `GET /ready` — liveness and readiness probes.
- `GET /metrics` — Prometheus metrics.

Design goals for the current CPU-only single-user deployment:

1. **Bounded concurrency** — one active generation at a time; a bounded waiting queue prevents
   unbounded work accumulation.
2. **Fail-closed admission** — oversized bodies, oversized prompts and explicit queue rejections are
   rejected before a backend generation starts.
3. **OpenAI wire-format compatibility** — OpenAI-style error envelopes, indexed streaming tool-call
   deltas, final usage chunks and context-limit errors.
4. **Reasoning control** — explicit `think` handling for Qwen3-family models so reasoning tokens do
   not silently consume the completion budget.
5. **Operational safety** — circuit breaker, backend timeouts, graceful shutdown, resource limits and
   Prometheus metrics.

## Architecture

```text
Jarvis CLI (local agent)           AI Stack (optional remote control plane)
          |                                      |
          +------------------+-------------------+
                             |
                             v
                     jarvis-inference
                     OpenAI-compatible API
                             |
                           Ollama
```

The dedicated inference VM owns model execution. Jarvis and AI Stack are independent sibling
consumers of the gateway; neither consumer runs Ollama or LiteLLM. Jarvis local execution must remain
usable when AI Stack is stopped. The gateway owns the model allowlist, bounded queue, concurrency,
backend timeouts and resource limits.

## Quick start

Requirements: Docker Engine, Docker Compose v2 and `curl`.

```bash
git clone https://github.com/abdullahalrifat/jarvis-inference.git
cd jarvis-inference
cp .env.example .env
chmod 600 .env
# Set INFERENCE_BIND_ADDRESS=0.0.0.0 and a long INFERENCE_API_KEY for remote access.
bash scripts/install.sh
```

The installer validates Docker/Compose, validates the Compose file, builds the gateway, starts
Ollama, ensures configured models are present, waits for readiness, and runs authenticated smoke
checks.

Verify the deployment:

```bash
docker compose ps
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/ready
curl -sS -H "Authorization: Bearer $INFERENCE_API_KEY" http://127.0.0.1:8080/v1/capabilities
```

## Install

For a dedicated inference VM serving either standalone Jarvis or AI Stack, set before install:

```env
INFERENCE_BIND_ADDRESS=0.0.0.0
INFERENCE_API_KEY=<long-random-secret>
```

Keep TCP 8080 restricted to the private network or trusted clients. Ollama port 11434 is never
published.

Install:

```bash
bash scripts/install.sh
```

## Configuration

All runtime settings come from environment variables (see [`.env.example`](.env.example) for the full
surface). Highlights:

| Variable | Default | Description |
| --- | --- | --- |
| `INFERENCE_HOST` | `0.0.0.0` | Gateway bind host. |
| `INFERENCE_PORT` | `8080` | Gateway bind port. |
| `INFERENCE_BIND_ADDRESS` | `127.0.0.1` | Host-side bind address; set to `0.0.0.0` for remote access. |
| `INFERENCE_API_KEY` | *(empty)* | Bearer token required by authenticated endpoints; required when remotely reachable. |
| `INFERENCE_ALLOW_INSECURE_NO_AUTH` | `false` | Fail-closed development escape hatch; production startup requires the key. |
| `MAX_CONCURRENT_REQUESTS` | `1` | Active inference slots. |
| `MAX_QUEUE_SIZE` | `8` | Waiting requests; the active generation is tracked separately. |
| `MAX_LOADED_MODELS` | `1` | Backend-side loaded models. |
| `MEMORY_BUDGET_GB` / `MEMORY_HEADROOM_GB` | `10` / `1.0` | Gateway memory budget and headroom. |
| `BACKEND_HTTP_TIMEOUT_SECONDS` | `600` | Backend (Ollama/llama.cpp) request timeout. |
| `QUEUE_TIMEOUT_SECONDS` | `90` | Waiting time before an unslotted request is rejected. |
| `EMBEDDING_TIMEOUT_SECONDS` | `30` | Embedding request timeout. |
| `CHAT_TIMEOUT_SECONDS` | `300` | Non-streaming chat timeout. |
| `STREAM_TIMEOUT_SECONDS` | `300` | Streaming read timeout. |
| `SHUTDOWN_TIMEOUT_SECONDS` | `30` | Graceful-shutdown bound for the scheduler. |
| `MODEL_REFRESH_SECONDS` | `15` | Model-state refresh cadence. |
| `CIRCUIT_FAILURE_THRESHOLD` / `CIRCUIT_RECOVERY_SECONDS` | `3` / `15` | Circuit breaker thresholds. |
| `OLLAMA_URL` | `http://ollama:11434` | Backend base URL. |
| `LLAMACPP_URL` | *(empty)* | Optional llama.cpp backend URL. |
| `OLLAMA_KEEP_ALIVE` | `30m` | Backend model residency. |
| `MAX_CONTEXT_LENGTH` | `8192` | Hard context cap applied before queue admission. |
| `MAX_EMBEDDING_BATCH_SIZE` | `64` | Inputs per embedding batch. |
| `INFERENCE_CPUS` / `INFERENCE_MEMORY_LIMIT` | `0.4` / `1g` | Gateway container CPU/memory caps (constrained-host profile). |
| `OLLAMA_CPUS` / `OLLAMA_MEMORY_LIMIT` | `1.5` / `8g` | Backend container CPU/memory caps (constrained-host profile). |
| `MIN_FREE_DISK_GIB` / `MIN_AVAILABLE_RAM_GIB` | `10` / `1` | Watchdog thresholds. |

`MAX_REQUEST_BODY_MIB` (default `8`) bounds request-body size and is enforced before validation;
prompts that do not fit the effective context are rejected with `CONTEXT_LIMIT_EXCEEDED` at admission.

## Production and constrained-host profiles

Recommended CPU-only VM for comfortable headroom:

- 4 vCPU
- 10-12 GiB RAM
- Ollama: up to 2.75 CPU / 9 GiB RAM
- Gateway: up to 0.50 CPU / 512 MiB RAM

Constrained 2-vCPU / 10-GiB hosts (such as a small homelab VM) should retain the lower defaults in
`.env.example`: Ollama 1.5 CPU / 8 GiB and gateway 0.4 CPU / 1 GiB. Do not copy the larger production
CPU limits onto a 2-vCPU VM without benchmarking and host-headroom checks.

The default `MAX_LOADED_MODELS=1` is intentionally conservative. A two-model setting may reduce cold
reloads when alternating between `qwen3:1.7b` and `nomic-embed-text`, but benchmark it on the actual VM
first; do not increase it blindly because `qwen3:4b` and context memory also compete for the 8-GiB
Ollama limit.

For CPU-only deployments, `MAX_CONTEXT_LENGTH` defaults to 8192 and request-level `num_ctx` overrides
above that cap are rejected before queue admission. `MAX_EMBEDDING_BATCH_SIZE` defaults to 64 inputs per
request to bound memory and latency; split larger ingestion jobs into smaller batches. Raise either
limit only after measuring peak RAM and throughput on the target VM.

Both profiles preserve:

- One active generation
- One loaded model
- Queue size 8 waiting requests (active generation is separate)
- Context length 8192
- Ollama keep-alive 30 minutes

Configured models:

- `qwen3:1.7b` — fast fallback model
- `qwen3:4b` — default model
- `qwen3:8b` — reasoning model
- `nomic-embed-text` — embeddings

## Queueing and CPU-only operation

The gateway intentionally runs one generation at a time on small CPU-only hosts. A bounded waiting
queue prevents unbounded work accumulation; `QUEUE_TIMEOUT_SECONDS` controls how long a request
waits for a slot (default: 90 seconds), while `MAX_QUEUE_SIZE` bounds waiting requests separately from
the active inference slot. Queue-depth telemetry counts waiting requests, not the active
generation. Queue timeout and queue-full responses are explicit pre-generation rejections and include
`Retry-After` guidance. Clients must not retry ambiguous generation/read timeouts because the backend
may already have executed the request.

Backend model-state refreshes occur only after a request owns the inference slot, avoiding redundant
concurrent refresh work while another generation is active. Increasing the queue deadline improves
tolerance to normal long generations; it does not increase throughput.

## Embedding efficiency

Embedding requests are sent to Ollama as a batch rather than one HTTP request per text. Response order
is preserved, incomplete batches fail explicitly, and an empty batch performs no backend request.
Batching reduces request overhead; it does not claim to eliminate embedding compute.

## Reasoning models

Qwen3-family models emit reasoning (`thinking`) by default, which silently consumes the completion
budget before any answer text. The gateway takes explicit control:

- `POST /v1/chat/completions` accepts `think: true|false`. Omitted requests default to
  `OLLAMA_DEFAULT_THINK` (default `false`). With thinking disabled, CPU decode throughput drops no
  tokens on reasoning and answers start immediately.
- When `think: true` is requested, reasoning is surfaced as `reasoning_content` (OpenAI-style) in
  non-streaming `choices[0].message` and in streaming deltas; `content` only fills for the
  post-thinking answer.
- CPU tune note: set `OLLAMA_NUM_THREADS` to the P-core count (e.g. 8 on an i7-14700). More threads
  slow decode down because throughput is memory-bandwidth bound. Measured: qwen3:4b ~26 tok/s,
  qwen3:8b ~14 tok/s at 8 threads.

## Deployment lifecycle

All commands below are run from the repository root on the dedicated inference VM.

### 1. First deployment

Install Docker Engine and Docker Compose v2 first, then:

```bash
git clone https://github.com/abdullahalrifat/jarvis-inference.git
cd jarvis-inference
cp .env.example .env
chmod 600 .env
```

For a dedicated inference VM serving Jarvis or AI Stack from another VM, configure:

```env
INFERENCE_BIND_ADDRESS=0.0.0.0
INFERENCE_API_KEY=<long-random-secret>
```

Keep TCP 8080 restricted to trusted private-network clients. Ollama port 11434 must remain internal.

Run the idempotent installer:

```bash
bash scripts/install.sh
```

Verify:

```bash
docker compose ps
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/ready
curl -sS -H "Authorization: Bearer $INFERENCE_API_KEY" http://127.0.0.1:8080/v1/capabilities
```

From a remote Jarvis/AI Stack VM:

```bash
curl -sS http://<INFERENCE_VM_IP>:8080/ready
```

### 2. Pull the latest changes

The installer does not replace your `.env`. Before updating:

```bash
cd ~/docker/jarvis-inference
git status
git pull --ff-only
```

If `git status` shows local source changes, stop and review them before pulling. Never use
`git reset --hard` unless you intentionally want to discard local changes.

### 3. Re-deploy after pulling changes

Use the same installer:

```bash
bash scripts/install.sh
```

It validates Compose, rebuilds the gateway, starts the services, ensures configured Ollama models
exist, waits for readiness, and runs smoke checks.

For a source-only gateway change, you can rebuild/recreate just the gateway:

```bash
docker compose build --pull jarvis-inference
docker compose up -d --force-recreate jarvis-inference
```

Do not remove the Ollama volume during normal redeployment; otherwise the models will need to be
downloaded again.

### 4. Normal operations

Check status:

```bash
docker compose ps
```

View gateway logs:

```bash
docker compose logs --tail=200 jarvis-inference
```

View Ollama logs:

```bash
docker compose logs --tail=200 ollama
```

Follow logs:

```bash
docker compose logs -f jarvis-inference
```

Restart only the gateway:

```bash
docker compose restart jarvis-inference
```

Restart both services:

```bash
docker compose restart ollama jarvis-inference
```

### 5. Shut down the inference stack

To stop the containers while preserving models and configuration:

```bash
docker compose stop
```

To stop and remove the containers/network while preserving the named Ollama model volume:

```bash
docker compose down
```

This is the preferred shutdown before VM maintenance or reboot.

Start again without rebuilding:

```bash
docker compose up -d
```

### 6. Full cleanup

For a normal application cleanup, use:

```bash
docker compose down
```

This removes containers and the Compose network but **does not remove the Ollama model volume**.

To deliberately remove the downloaded models as well:

```bash
docker compose down -v
```

This is destructive: all Ollama model data in the Compose volume will be deleted and the next
deployment will download the models again.

Do not run `docker system prune --volumes` as part of routine maintenance because it can remove
unrelated Docker resources.

### 7. Complete removal from the VM

If the inference VM is being retired and you want to remove the application, containers, network,
and model volume:

```bash
cd ~/docker/jarvis-inference
docker compose down -v
cd ..
rm -rf jarvis-inference
```

Back up the production `.env` securely before doing this if the VM may need to be rebuilt.

### 8. Recovery after a VM reboot

Docker should restart the Compose services according to the Compose restart policy. Verify:

```bash
cd ~/docker/jarvis-inference
docker compose ps
curl -sS http://127.0.0.1:8080/ready
```

If the stack did not start:

```bash
docker compose up -d
```

If the gateway image or source needs rebuilding:

```bash
bash scripts/install.sh
```

### 9. Troubleshooting remote connection refused

On the inference VM:

```bash
docker compose ps
hostname -I
```

The gateway should show a publication similar to:

```text
0.0.0.0:8080->8080/tcp
```

Check the bind setting:

```bash
grep '^INFERENCE_BIND_ADDRESS=' .env
```

For cross-VM access it should be:

```env
INFERENCE_BIND_ADDRESS=0.0.0.0
```

Then recreate the gateway:

```bash
docker compose up -d --force-recreate jarvis-inference
```

From the client VM:

```bash
curl -sS http://<INFERENCE_VM_IP>:8080/ready
```

If local readiness works but remote access fails, check the VM firewall and private-network routing.
Do not expose port 11434.

## Operations

From the repository root:

```bash
docker compose ps
docker compose logs --tail=100 jarvis-inference
docker compose logs --tail=100 ollama
docker compose restart jarvis-inference
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

For a remote client, use the VM's private IP instead of 127.0.0.1 and include the inference API key
for authenticated endpoints.

Upgrade the application:

```bash
git pull --ff-only
bash scripts/install.sh
```

This rebuilds the gateway while preserving the Ollama model volume.

## Model integrity

`models/manifest.yaml` is the deployment model record. Model digest changes should be reviewed
deliberately rather than silently accepted.

The installer ensures the configured model names exist, but it does not overwrite the manifest digest
automatically.

## Security

- Loopback binding is the default.
- Remote binding is explicit through `INFERENCE_BIND_ADDRESS`.
- Remote deployments require `INFERENCE_API_KEY`.
- Ollama is not exposed outside the Docker network.
- Gateway runs non-root with a read-only filesystem and no-new-privileges.
- Production container images are pinned by digest.
- Docker logs have bounded rotation.
- CI includes tests, security scanning and release checks.

## Development

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --cov=inference --cov-report=term-missing --cov-fail-under=70
```

## Operational validation

A target-host validation surface is available under `scripts/world_class_check.sh`, `scripts/benchmark.sh`
and `scripts/soak.sh`. Production certification requires retained benchmark/soak evidence on the actual
inference VM; CI alone is insufficient.

## Troubleshooting

| Symptom | Likely cause / check |
| --- | --- |
| `docker compose ps` shows `jarvis-inference` unhealthy | Gateway healthcheck probes `/ready`, which is backend-aware; check Ollama first with `docker compose logs ollama`. |
| `429` / queue rejection with `Retry-After` | Queue timeout or queue-full; `QUEUE_TIMEOUT_SECONDS` / `MAX_QUEUE_SIZE` are explicit pre-generation rejections. |
| `400 CONTEXT_LIMIT_EXCEEDED` | Prompt exceeds the effective context window; reduce context or raise `MAX_CONTEXT_LENGTH` deliberately. |
| `413 REQUEST_TOO_LARGE` | Request body exceeds `MAX_REQUEST_BODY_MIB`; split large payloads. |
| Backend connection errors | Circuit breaker tripped (`CIRCUIT_FAILURE_THRESHOLD`); recovery after `CIRCUIT_RECOVERY_SECONDS`. |
| Auth failures on smoke checks | Set `INFERENCE_API_KEY` and send `Authorization: Bearer ...`; remote binding requires the key. |

## Contributing

Contributions are welcome. Before opening a pull request:

1. Search existing issues and pull requests.
2. Open an issue for large API or behavioral changes.
3. Keep the gateway provider-neutral and dependency-light at runtime.
4. Add or update tests for behavior changes.

Before submitting a PR, run the development checks above plus `bash -n scripts/*.sh` and
`python scripts/write_certification_manifest.py certification-evidence.json` for the CI-equivalent
local gate.

## Support

Use GitHub Issues for reproducible bugs and focused feature requests, and GitHub Discussions for
questions and design discussion. Before opening an issue, read the README and the latest changelog, and
include your Python version, operating system, package version, and a minimal reproduction. Remove API
keys and sensitive trace content.

## License

Released under the [MIT License](LICENSE).
