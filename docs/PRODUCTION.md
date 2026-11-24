# Production deployment

## Dedicated inference VM

Recommended CPU-only profile:

- 4 vCPU
- 10-12 GiB RAM
- llama-server: `LLAMACPP_THREADS` CPU / `MEMORY_BUDGET_GB` RAM
- Gateway: `INFERENCE_CPUS` CPU / `INFERENCE_MEMORY_LIMIT` RAM
- SSD-backed `./models` mount containing the production GGUF
- One active generation and one loaded model

## Install

```bash
git clone https://github.com/Akr-Byte/jarvis-inference.git
cd jarvis-inference
cp .env.example .env
cp /path/to/production-model.gguf models/
```

For cross-VM access:

```env
INFERENCE_BIND_ADDRESS=0.0.0.0
INFERENCE_API_KEY=<long-random-secret>
INFERENCE_ALLOW_INSECURE_NO_AUTH=false
```

Restrict TCP 8080 to the private network or trusted client IPs. The llama-server port 8081 is bound
to loopback because it has no authentication; do not publish it on a public interface.

Run:

```bash
bash scripts/install.sh
```

The installer validates the deployment, checks for a mounted GGUF, builds the gateway and
llama-server image, starts both services, waits for readiness, and runs smoke checks. Set a non-empty
`INFERENCE_API_KEY` before deploying; the service refuses startup without one unless the explicit
development-only override is enabled.

## Operations

```bash
docker compose ps
docker compose logs --tail=100 jarvis-inference
docker compose logs --tail=100 llama-server
docker compose restart jarvis-inference
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

Upgrade:

```bash
git pull --ff-only
bash scripts/install.sh
```

Mounted GGUF files are preserved across image rebuilds. The standalone Ollama profile remains
available through `docker-compose.ollama.yml`.

## Resource policy

| Resource | Production budget |
| --- | ---: |
| VM CPU | 4 vCPU |
| VM RAM | 10-12 GiB |
| llama-server CPU | `LLAMACPP_THREADS` (quota from Compose `cpus`) |
| llama-server RAM | `MEMORY_BUDGET_GB` (Compose `mem_limit`) |
| Gateway CPU | `INFERENCE_CPUS` |
| Gateway RAM | `INFERENCE_MEMORY_LIMIT` |
| Generation concurrency | `MAX_CONCURRENT_REQUESTS` (keep <= `LLAMACPP_PARALLEL`) |
| Queue | `MAX_QUEUE_SIZE` |
| Loaded models | 1 |
| Per-slot context | `MAX_CONTEXT_LENGTH` |
| Total engine context | `MAX_CONTEXT_LENGTH * LLAMACPP_PARALLEL` |

## Model integrity

`models/manifest.yaml` records production model identities and reviewed digests. Do not silently change a recorded digest.

The installer verifies that at least one GGUF is mounted but never replaces model files. Model
digest changes should be reviewed separately before updating the manifest.

## Security

- Loopback binding is the default.
- Remote binding is explicit.
- Startup fails closed if `INFERENCE_API_KEY` is empty, unless `INFERENCE_ALLOW_INSECURE_NO_AUTH=true` is explicitly selected for isolated local development.
- Remote deployments require API-key authentication.
- llama-server port 8081 is published on loopback only; it has no authentication.
- The gateway is the only authenticated client entry point.
- Both containers run with no-new-privileges; the gateway is read-only.
- Container images are pinned by digest.
- Keep port 8080 private and do not expose metrics publicly.
