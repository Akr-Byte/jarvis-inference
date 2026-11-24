# Runbook

## Health

From the inference VM:

```bash
docker compose ps
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
```

From a client VM:

```bash
curl http://<INFERENCE_IP>:8080/ready
```

## Restart

Restart only the gateway when the application is unhealthy:

```bash
docker compose restart jarvis-inference
```

Restart the engine when llama-server is unhealthy:

```bash
docker compose restart llama-server
```

Restart both services:

```bash
docker compose restart llama-server jarvis-inference
```

## llama.cpp engine and model switching

The default Compose profile runs llama-server in its own container. The binary is installed in the
image; only `./models` is mounted into the container as `/models:ro`. Check the engine and gateway:

```bash
docker compose ps
curl -sS http://127.0.0.1:8081/health
curl -sS http://127.0.0.1:8080/ready
docker compose logs --tail=100 llama-server
```

`llama-server` loads exactly one GGUF when its process starts. The gateway lifecycle endpoints
(`/v1/models/{id}/load` and `/unload`) are disabled in this profile because `LLAMACPP_MANAGED=false`
is set for the gateway, so **changing models requires restarting the engine container**. The
gateway normally does not need a restart as long as the public alias stays the same.

1. Put the new GGUF in `./models/`:

   ```bash
   cp /path/to/new-model.gguf ./models/
   ```

2. Pin the file when several GGUFs are mounted. Compose forwards `LLAMA_MODEL`; otherwise the
   entrypoint selects the first path in sorted order and warns:

   ```bash
   export LLAMA_MODEL=/models/new-model.gguf
   ```

3. Recreate the engine, then verify:

   ```bash
   docker compose up -d --force-recreate llama-server
   curl -sS http://127.0.0.1:8081/health
   curl -sS http://127.0.0.1:8080/ready
   ```

If the alias also changes, update `LLAMACPP_ALIAS`, `MODEL_REGISTRY` and `DEFAULT_MODEL` in `.env`
and recreate both services:

```bash
docker compose up -d --force-recreate llama-server jarvis-inference
```

Native managed mode is still available for workstation deployments that set
`LLAMACPP_MANAGED=true` and let the gateway supervise its own llama-server child. In that mode the
authenticated `/v1/runtime/status`, `/v1/runtime/capabilities` and model load/unload endpoints can
switch among configured `LLAMACPP_MODEL_MAP` profiles without restarting the gateway.

## Performance verification

Tuning only counts if it improves the deployed profile. Run the benchmark before and after a change:

```bash
bash scripts/benchmark.sh
```

It uses `DEFAULT_MODEL` from `.env` and reports median total latency, time to first token and decode
tokens/s. Useful engine-side checks:

```bash
# llama-server Prometheus metrics are enabled by LLAMACPP_METRICS=true.
curl -sS http://127.0.0.1:8081/metrics | grep -E 'llama:|cache|prompt|tokens' | head -50

# Effective runtime flags in the container process list.
docker compose exec llama-server ps -o args= -p 1
```

Compare `decode_tokens_per_s` after changing threads, draft models, KV quantization or cache reuse.
Keep `MAX_CONCURRENT_REQUESTS <= LLAMACPP_PARALLEL`; extra gateway concurrency only queues inside
llama-server. Quantized KV types (`q8_0`, `q4_0`, ...) require flash attention; the container
entrypoint enables it automatically and skips unsupported flags with a warning.

## Update

```bash
git pull --ff-only
bash scripts/install.sh
```

The installer rebuilds the gateway and llama-server image. GGUF files in `./models` are preserved.

## Networking

If clients receive connection refused, verify:

```bash
docker ps
hostname -I
```

The gateway must publish 8080 as `0.0.0.0:8080->8080/tcp` for cross-VM access. Set `INFERENCE_BIND_ADDRESS=0.0.0.0` in `.env`.

Keep the host firewall restricted to trusted private-network clients.

## Standard lifecycle

### Deploy from scratch

```bash
git clone https://github.com/Akr-Byte/jarvis-inference.git
cd jarvis-inference
cp .env.example .env
chmod 600 .env
# copy at least one GGUF into ./models
cp /path/to/model.gguf models/
# edit .env; for cross-VM access set INFERENCE_BIND_ADDRESS=0.0.0.0 and a real INFERENCE_API_KEY
bash scripts/install.sh
```

### Update and redeploy

```bash
cd ~/docker/jarvis-inference
git status
git pull --ff-only
bash scripts/install.sh
```

The installer is the canonical redeployment path. It preserves GGUF files in `./models`.

### Stop / start

```bash
docker compose stop
docker compose start
```

### Shutdown

```bash
docker compose down
```

This removes containers and the Compose network but keeps model data.

### Destructive cleanup

```bash
docker compose down -v
```

This removes named volumes. Mounted GGUF files under `./models` are host files and are not removed; Ollama model volumes, if used separately, would be removed.

### Complete application removal

```bash
docker compose down -v
cd ..
rm -rf jarvis-inference
```

Only do this when retiring the VM/application. Preserve the production `.env` securely if recovery is required.

### Verify after reboot

```bash
docker compose ps
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/ready
```

If services did not start automatically:

```bash
docker compose up -d
```
