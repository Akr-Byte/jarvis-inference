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

Restart the full stack when Ollama itself is unhealthy:

```bash
docker compose restart ollama jarvis-inference
```

## Native llama.cpp runtime

The default Compose profile still uses Ollama. For native llama.cpp, follow the workstation
profile in the README and set `LLAMACPP_MANAGED=true`. Check the managed child and probed options
with authenticated requests:

```bash
curl -H "Authorization: Bearer $INFERENCE_API_KEY" http://127.0.0.1:8080/v1/runtime/status
curl -H "Authorization: Bearer $INFERENCE_API_KEY" http://127.0.0.1:8080/v1/runtime/capabilities
tail -n 100 logs/llama-server.log
```

`state=failed` includes the latest startup/recovery error. Check the GGUF path, model context
metadata, estimated KV memory and available RAM before lowering headroom or raising the context.
An unmanaged process occupying the configured port is never adopted or signaled; stop it through
the supervisor that launched it or choose a free port. Model load/unload requests are rejected
while the queue or active-request set is nonempty. External-server mode reports lifecycle controls
as unsupported.

## Update

```bash
git pull --ff-only
bash scripts/install.sh
```

The installer rebuilds the gateway and preserves the Ollama model volume.

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
git clone https://github.com/abdullahalrifat/jarvis-inference.git
cd jarvis-inference
cp .env.example .env
chmod 600 .env
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

The installer is the canonical redeployment path. It preserves the Ollama named volume.

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

This also removes the Ollama model volume. Models must be downloaded again on the next deployment.

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
