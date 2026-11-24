# Changelog

## Unreleased

## 0.6.0 - 2026-11-24

### Added

- Optional native `llama-server` supervision with ownership/PID reporting, health gating, bounded
  restart recovery, graceful shutdown, live log rotation, and version-probed optional flags.
- GGUF metadata inspection, complete split-shard discovery, explicit model maps, model lifecycle
  endpoints, runtime capabilities, context/output admission, and estimated weights-plus-KV memory
  checks for managed local runtimes.
- Workstation and constrained homelab environment profiles. Existing Ollama defaults remain in
  place.
- Qwen3-Next hybrid attention/recurrent memory estimation, component-level managed-runtime
  admission diagnostics, tokenizer-backed llama.cpp chat context checks, and startup/shutdown race
  protection for the owned process group.
- Containerized `llama-server` image with the official llama.cpp release binary installed at build
  time. No host binary or shared-library mounts are required; only `./models` is mounted read-only.
- Deterministic GGUF selection with `LLAMA_MODEL`, model aliasing with `LLAMA_ALIAS`, and a startup
  loop that waits for a mounted model instead of entering a restart loop.
- Documented containerized model switching: restart `llama-server` to load a different GGUF; the
  gateway refreshes the catalog automatically when the alias is unchanged.

### Changed

- Managed llama.cpp now passes the engine-parity flags that other OpenAI-compatible servers rely
  on: `--jinja` for the model chat template (tool calls, roles, reasoning extraction), `--metrics`
  for engine-level telemetry, plus opt-in `--mlock`, `--n-gpu-layers`, `--cache-reuse` and
  `--reasoning-format`. Every flag is still gated on the `--help` capability probe.
- CPU thread defaults follow the physical core count instead of a fixed `8`, avoiding SMT
  oversubscription on small hosts.
- `LLAMACPP_STARTUP_TIMEOUT_SECONDS` defaults to `300` so large GGUFs can finish loading.
- Startup logs warn about configurations that degrade throughput or safety: gateway concurrency
  above `LLAMACPP_PARALLEL`, unused llama.cpp slots, quantized KV caches without flash attention,
  non-loopback managed binds, and `LLAMACPP_JINJA=false`.
- Empty numeric environment values now fall back to defaults instead of raising at import time.
- The default Compose stack now runs llama.cpp as the primary engine in a dedicated container. The
  gateway connects over `LLAMACPP_URL` and overrides managed mode to `false`; native managed mode
  remains available for workstation deployments.
- The gateway image installs its pinned Python dependencies and starts through `uvicorn`, so the
  Compose build is self-contained.
- The installer, environment example, README and runbook now describe the llama.cpp container
  workflow, including how to change the mounted model.
- Release metadata (version, changelog top entry, README status line and workflow Dockerfile path) is
  now checked by `scripts/check_release_consistency.py` in both CI and release gates. The stale PyPI
  badge was removed and links now use the canonical `Akr-Byte` repository.
- Tag-triggered releases are idempotent and support a manual retry, validate distributions with
  `twine`, verify the wheel in a clean environment, attach `SHA256SUMS`, publish lowercase GHCR tags
  including `latest`, and scan the image at `HIGH,CRITICAL` to match the security workflow.
- `docs/releasing.md` documents the version policy, pre-release checklist, local CI-equivalent gate,
  publication steps and post-release verification.
- The gateway image installs the patched `zlib1g` build and drops pip/ensurepip after dependency
  installation. This clears the `HIGH` Trivy findings from the pinned base's stale `zlib1g` and
  from pip's vendored SBOM (stale `msgpack`, `setuptools` and `urllib3` copies that are never used
  at runtime), without suppressing any advisory.
- The release workflow refuses to publish a tag whose commit is not an ancestor of `origin/main`,
  so a tag pushed from an unmerged feature branch can no longer produce a release.
- The `llama-server` bind mount now follows `MODEL_DIRECTORY`, so GGUF files can live in the OS home
  directory (`~/models`) or any absolute host path instead of the repository checkout. The installer
  expands `~` the same way and all model-switching docs use the configured host directory. The
  installer now discovers GGUFs one directory deep, matching the engine entrypoint's split-shard
  folders. `.env.example` and the configuration table now document `LLAMA_MODEL` separately from the
  native-mode `LLAMACPP_MODEL`, since the containerized engine needs an in-container path.

### Performance

- Optional speculative decoding: `LLAMACPP_DRAFT_MODEL` (draft GGUF path) now maps to the
  llama-server `--draft-model` flag when the installed build advertises it, enabling the
  draft-then-verify loop that is the largest CPU-only throughput lever for local serving.
- Optional speculative-decoding sampling bounds: `LLAMACPP_DRAFT_MAX_LEN` (draft token horizon,
  default 5) and `LLAMACPP_DRAFT_MIN_LEN` (minimum draft length before verification starts, default
  5) map to llama-server `--draft-max-len` / `--draft-min-len` when a draft model is configured.
- Optional speculative-decoding draft context size: `LLAMACPP_DRAFT_CTX_SIZE` (default 0 = engine
  default) maps to `--draft-ctx-size` when a draft model is configured and the flag is supported.
- Exact tokenizer-based context admission only runs near the context limit (>= 80% of prompt plus
  output budget); shorter prompts skip the extra engine round-trip before generation.
- Streaming requests always ask llama.cpp for a terminal usage chunk so completion tokens are
  metered, and the usage-only chunk is filtered out for clients that did not request it.
- GGUF discovery caches parsed headers by path, size and mtime and skips materialising tokenizer
  vocabulary arrays, so `/v1/model-registry`, `/v1/models/{id}` and status endpoints no longer
  re-parse every model on each call (a 150k-token vocabulary costs ~20 ms per scan).
- `inference_active_model` is counted per model, so it stays correct when
  `MAX_CONCURRENT_REQUESTS > 1`.
- The body-limit middleware is pure ASGI: no per-request wrapper task, and streaming responses
  keep their disconnect semantics.
- llama.cpp HTTP connection limits scale with gateway concurrency instead of a fixed pool of two.
- The containerized llama-server now receives the full `LLAMACPP_*` tuning set from `.env`. The
  entrypoint capability-probes renamed flags, sizes context as
  `MAX_CONTEXT_LENGTH * LLAMACPP_PARALLEL`, enables flash attention for quantized KV caches, and
  supports speculative decoding on both current `--spec-draft-*` and legacy `--draft-*` builds.
- Thread defaults are clamped by cgroup CPU quota and process affinity, avoiding SMT oversubscription
  inside resource-limited containers.
- Oversized multi-message chats are compacted once by dropping the oldest non-system messages and
  the compacted payload is actually forwarded to the engine; the previous implementation discarded
  its trimmed copy and re-estimated every suffix.
- llama.cpp model-state refreshes reuse the `/v1/models` response instead of issuing a second
  identical request for `loaded_models()`.
- Stale stream slots are reclaimed at 1.1x the configured timeout with a 5-30s sweep, instead of
  1.5x with a 150s sweep that could stall the only slot for minutes.
- `scripts/benchmark.sh` reads `DEFAULT_MODEL` from `.env` and reports time to first token and
  decode tokens/s in addition to total latency.
- SSE lines are decoded once by the scheduler and the parsed event is attached to the streamed line,
  so the API layer no longer re-parses every token to decide whether to suppress a usage chunk.
- The containerized engine now receives explicit `--threads` / `--threads-batch` values matching its
  Compose CPU quota, instead of inheriting host-core defaults that oversubscribe the container.
- Model-discovery results are cached for two seconds, removing repeated `rglob` and `stat` work from
  status and metadata endpoints while still noticing model changes.

### Fixed

- Managed model load pre-flights the target GGUF and returns `409 MODEL_FILE_MISSING` without
  unloading the running model, instead of stopping the working model before discovering that the
  new profile path is missing.
- Gateway concurrency above `LLAMACPP_PARALLEL` is logged at startup; those requests previously
  queued inside `llama-server` and surfaced as gateway timeouts rather than a clear `429`.
- `llama-server` healthchecks use `CMD-SHELL`, and the entrypoint passes an explicit `--port 8080`
  and the configured model alias, so Compose dependency gating and gateway model discovery work
  reliably.
- `MAX_CONTEXT_LENGTH` now defaults to the documented 16384 tokens in code and environment examples;
  llama-server receives an explicit `--ctx-size` matching the gateway admission limit.
- `Settings.active_backend` now reports `ollama` when no llama.cpp URL or managed child is configured
  instead of always reporting `llama.cpp`.
- GGUF selection is sorted and warns when multiple models are mounted; the engine port is published
  on loopback and the engine container has CPU, memory and PID limits.
- `LLAMACPP_CACHE_RAM_MIB` is passed to `llama-server --cache-ram` in MiB rather than bytes, matching
  the engine's documented unit.
- Speculative-decoding sampling flags are only forwarded when `LLAMACPP_DRAFT_MODEL` is set, so a
  target-only engine never receives draft parameters.
- The installer waits up to six minutes for `llama-server` health, matching the extended Compose
  startup window for large GGUF loads.
- Managed `--cache-ram` now receives MiB rather than bytes, matching the pinned llama.cpp build, and
  zero is forwarded explicitly so the engine's 8192 MiB default cannot silently apply.
- A compacted prompt that was rejected by the exact tokenizer is re-counted with the same tokenizer
  before being accepted; a heuristic can no longer re-admit a prompt the engine rejected.
- Stale-stream cleanup cancels the task driving the abandoned backend stream and marks the lease
  released, preventing a late cleanup from decrementing another live request's model gauge.
- Engine capability probing matches whole option tokens, so a longer or deprecated flag in the help
  text can no longer be mistaken for support.
- The benchmark's decode-rate metric excludes the first token, whose generation time is already
  counted in time to first token.
- Enabling `LLAMACPP_EMBEDDINGS` with an external engine now logs a warning that `--embedding` must
  be supplied through `LLAMA_EXTRA_ARGS`; the thread-oversubscription warning is limited to managed
  mode, where gateway affinity actually describes the engine process.
- All three images now declare a non-root `USER`, clearing Trivy `DS-0002` in the security workflow;
  the gateway, engine and Ollama containers were verified to start and serve as `appuser`.
- The release workflow builds the root `Dockerfile` instead of the deleted `docker/Dockerfile`, so
  tag-triggered image publishing works with the modular image layout.
- Ruff no longer flags the forward reference in the discovery cache annotation, and the formatter
  check passes across the tree.

## 0.5.1

### Fixed

- **GW-15 wiring**: the scheduler now passes `trial_timeout` to every circuit
  breaker and calls `abandon_trial()` when an admitted half-open trial exits
  without a verdict (cancellation, client disconnect, or a pre-backend
  admission failure). Previously the trial-wedge protection existed in
  `CircuitBreaker` but was never enabled, so a half-open trial that exited
  without reporting success/failure could reject every caller forever.
- **GW-21 llama.cpp streaming**: OpenAI-format streams from llama.cpp end with
  the `data: [DONE]` sentinel rather than Ollama's `done` event. The scheduler
  now treats `[DONE]` (and an OpenAI `usage` chunk) as the terminal event
  instead of failing every llama.cpp stream with `502 STREAM_INCOMPLETE`, and
  token metrics are recorded from the OpenAI usage chunk. The gateway no
  longer appends a duplicate `[DONE]` when the backend already sent one.
- **GW-5 chunked bodies**: `POST /v1/*` requests using
  `Transfer-Encoding: chunked` without a `content-length` are rejected with
  `411 LENGTH_REQUIRED` instead of bypassing the request-size cap.
- **GW-13 request ids**: client-supplied `X-Request-ID` values are only echoed
  when they are at most 128 header-safe characters; otherwise a generated id
  is used. The previous behavior echoed arbitrary/unbounded values into
  response headers and payload `id` fields.
- **GW-11 tool_choice honesty**: `tool_choice=none` is now enforced by
  stripping the tools; other values are tolerated but explicitly advertised as
  unenforced via `/v1/capabilities.limitations`.
- Embeddings responses report an estimated `usage.prompt_tokens` instead of a
  misleading zero.
- `LlamaCppBackend.available_models()` degrades to an empty catalog on
  transport errors instead of leaking a raw `httpx` exception through the
  readiness path.

## 0.5.0

### Breaking / wire-contract changes

- InferenceError responses now return the OpenAI error envelope
  (`{"error": {"code", "message", "retryable", "request_id"}}`) instead of
  FastAPI's `{"detail": ...}` shape, so OpenAI-SDK clients get native error
  handling across non-streaming and streaming paths.
- Previously tolerated-but-unimplemented chat parameters (`n`, `logprobs`,
  `top_logprobs`, `logit_bias`, `presence_penalty`, `frequency_penalty`) are
  now rejected with an OpenAI-format `400 UNSUPPORTED_PARAMETER` instead of
  being silently dropped.
- A backend stream that ends without its terminal `done` event now fails with
  `502 STREAM_INCOMPLETE` (retryable) instead of being counted as a success.
- `POST /v1/*` request bodies larger than `MAX_REQUEST_BODY_MIB` (default 8)
  are rejected with `413 REQUEST_TOO_LARGE` before validation.
- Prompts that do not fit the effective context window are rejected with
  `400 CONTEXT_LIMIT_EXCEEDED` at admission instead of being silently
  truncated by the backend with HTTP 200.

### Streaming compliance

- Malformed backend SSE lines are skipped and logged instead of aborting an
  otherwise healthy stream.
- Ollama's `done_reason` is mapped exactly like the non-streaming path, so
  tool calls finish as `finish_reason: "tool_calls"` in streaming too.
- Streaming tool calls are emitted as OpenAI-indexed deltas
  (`{index, id, type, function:{name, arguments}}`) so OpenAI-SDK stream
  parsers work.
- `stream_options.include_usage` is honored with a final usage-only chunk
  (empty choices); without it the usage object stays inline on the final
  chunk.

### Ops / reliability

- `MAX_QUEUE_SIZE=0` is clamped to 1 (`asyncio.Queue` treats maxsize=0 as
  unbounded), restoring the bounded-queue guarantee.
- uvicorn starts with `--timeout-graceful-shutdown 25` (under the Compose
  30s `stop_grace_period`) so in-flight generations finish or fail cleanly
  instead of being killed mid-stream.
- The ollama container now receives the recorded `OLLAMA_NUM_THREADS` /
  `OLLAMA_KV_CACHE_TYPE` operator knobs.
- Compose and Dockerfile healthchecks probe `/ready` (backend-aware) instead
  of `/health`.
- Falsy `keep_alive` (0 = unload immediately) no longer falls back to the
  configured residency default; the `think` flag is only sent to
  thinking-capable model families.
- `ACTIVE_REQUEST_AGE` tracks the oldest active entry instead of being
  stomped to zero on every admission.
- The OpenAPI version metadata now tracks the installed package version
  instead of a hardcoded constant.

## 0.4.0

### Features

- First-class reasoning-model support: `/v1/chat/completions` accepts a `think` parameter (`true`/`false`). Qwen3-family models are forced to explicit non-thinking mode by default (`OLLAMA_DEFAULT_THINK`, default `false`) so callers no longer silently burn the token budget on reasoning.
- Reasoning output is surfaced OpenAI-style as `reasoning_content` in both non-streaming messages and streaming deltas when thinking is enabled, instead of empty content blocks.

### Ops notes

- Optimized CPU profile on modern hosts: measured ~26 tok/s on `qwen3:4b` and ~14 tok/s on `qwen3:8b` with `OLLAMA_NUM_THREADS=8` on an i7-14700; thread counts beyond 8 collapse restartless throughput because decode is memory-bandwidth bound.
- Model registry now provisioned with `qwen3:8b`; recorded in `models/manifest.yaml`.

## 0.3.3

### Reliability

- Acquire the bounded inference slot before refreshing backend model state, preventing concurrent queued requests from duplicating model-state checks; release the slot even if preparation fails.
- Count only waiting requests in queue depth/capacity; the active inference slot is reported separately.
- Reuse cached model discovery during readiness probes instead of forcing redundant backend calls, and align OpenAPI version metadata with package version 0.3.3.
- Honor `MAX_LOADED_MODELS` in Compose while preserving the safe default of one model; this enables measured two-model residency experiments without changing default memory behavior.
- Bound per-request context overrides and embedding batch size before queue admission to protect CPU-only hosts from oversized requests.
- Include timed-out requests in queue-wait latency histograms so observed wait distributions are not biased toward successful admissions.
- Increase the default queue wait to 90 seconds to match realistic CPU-only generation latency while retaining bounded queue size and single-request concurrency.
- Return `Retry-After` guidance for explicit queue-full and queue-timeout rejections, including streaming requests before SSE headers are committed.

### Tests

- Cover serialized model preparation and queue/slot cleanup after a waiting request times out.


## 0.3.2

### Performance and reliability

- Batch embedding inputs into one Ollama `/api/embed` request, preserving input order and rejecting incomplete batches.
- Return immediately for an empty embedding batch without waking the backend.

### Tests

- Cover batching, response ordering, empty input, and incomplete backend responses.


## 0.3.1

### Security

- Fail startup when `INFERENCE_API_KEY` is missing unless the explicit development-only `INFERENCE_ALLOW_INSECURE_NO_AUTH=true` override is set.
- Return 503 when authentication is not configured and 401 when configured credentials are missing or invalid.

### Reliability

- Add regression coverage ensuring ambiguous backend generation timeouts are not automatically replayed.
- Add security configuration and authentication tests.
