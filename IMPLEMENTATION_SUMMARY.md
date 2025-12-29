# MemOpt Two-Tier Implementation Summary

## Overview

Implemented a secure two-tier architecture separating control plane (Coolify) from data plane (GPU worker).

## Architecture

```
Customer → Control Plane (API auth/billing) → Worker (GPU inference)
            (Coolify :3000)                    (GPU Server :8001)
            Public API                          Internal only
            Bearer <API_KEY>                    X-Worker-Token
```

## Files Created

### Worker Service (`worker/`)

1. **worker/main.py** - Secure FastAPI worker service
   - Token authentication via `X-Worker-Token` header
   - Model loading at startup
   - `/health` endpoint (no auth)
   - `/generate` endpoint (requires token)
   - Request ID propagation for tracing

2. **worker/requirements.txt** - Worker dependencies
   - torch, transformers, accelerate (MemOpt core)
   - fastapi, uvicorn, pydantic (API)

3. **worker/Dockerfile** - Worker container
   - Based on `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
   - Supports GPU and CPU modes
   - Non-root user (memopt:1000)
   - Health check configured

4. **worker/docker-compose.yml** - Worker deployment
   - GPU support (commented by default)
   - Volume for HuggingFace model cache
   - Environment variables from .env

5. **worker/.env.example** - Configuration template
   - WORKER_TOKEN (required, 32+ chars)
   - MODEL_NAME (required)
   - OPTIMIZATION_LEVEL, HF_TOKEN (optional)

### Documentation

1. **DEPLOYMENT.md** - Complete production deployment guide
   - Step-by-step instructions
   - Security configuration
   - Troubleshooting
   - Production checklist

2. **QUICKSTART.md** - Quick reference guide
   - 10-minute deployment
   - Common commands
   - Key environment variables

3. **IMPLEMENTATION_SUMMARY.md** - This file

## Files Modified

### Control Plane

1. **saas/config.py**
   - Added `WORKER_URL` (required in production)
   - Added `WORKER_TOKEN` (required in production, 32+ chars)
   - Added `WORKER_TIMEOUT_SECONDS` (default: 120)
   - Validators ensure fail-fast startup if missing in production

2. **saas/main.py**
   - Updated `/v1/infer` to call worker service
   - Sends `X-Worker-Token` header (NOT customer API key)
   - Propagates `request_id` for tracing
   - Handles worker auth failures (401)
   - Updated `/health` to check worker connectivity
   - Removed local model caching (now on worker)

3. **saas/requirements.txt**
   - Already had `httpx==0.27.0` (no change needed)

## Security Features

1. **Token Authentication**
   - Worker requires `X-Worker-Token` header
   - Rejects requests without valid token (401)
   - Token must be 32+ characters
   - Same token configured in both worker and control plane

2. **Isolation**
   - Customer API keys NEVER sent to worker
   - Worker only accessible from control plane IP
   - Firewall rules enforce network isolation

3. **Request Tracing**
   - Control plane generates `request_id`
   - Propagated to worker for correlation
   - Logged at both tiers

## Environment Variables

### Control Plane (Coolify)

```bash
# Existing
DATABASE_URL=postgresql://...
ADMIN_API_KEY=...
ENV=production

# New (required in production)
WORKER_URL=http://GPU_SERVER_IP:8001
WORKER_TOKEN=<32+ char token>

# New (optional)
WORKER_TIMEOUT_SECONDS=120
```

### Worker (GPU Server .env)

```bash
# Required
WORKER_TOKEN=<same as control plane>
MODEL_NAME=meta-llama/Llama-2-7b-hf

# Optional
OPTIMIZATION_LEVEL=7
WORKER_PORT=8001
HF_TOKEN=<for gated models>
```

## API Flow

### Request Flow

1. Customer → `POST /v1/infer` with `Authorization: Bearer <API_KEY>`
2. Control plane validates API key, checks quotas
3. Control plane → `POST /generate` to worker with `X-Worker-Token`
4. Worker validates token, runs inference
5. Worker → returns result with token counts
6. Control plane logs usage, returns to customer

### Health Check Flow

1. Customer/Monitor → `GET /health`
2. Control plane checks database, redis, worker
3. Control plane → `GET /health` to worker (no auth)
4. Returns overall status

## Testing Commands

### Worker Health (from GPU server)

```bash
curl http://localhost:8001/health
```

### Worker Health (from Coolify VPS)

```bash
curl http://GPU_SERVER_IP:8001/health
```

### Worker Inference (with token)

```bash
curl -X POST http://localhost:8001/generate \
  -H "X-Worker-Token: $WORKER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-123",
    "prompt": "What is AI?",
    "model": "meta-llama/Llama-2-7b-hf",
    "max_tokens": 50,
    "temperature": 0.7,
    "optimization_level": 7
  }'
```

### Control Plane Health

```bash
curl https://memopt.sophisticatesai.com/health
```

### End-to-End Inference

```bash
curl -X POST https://memopt.sophisticatesai.com/v1/infer \
  -H "Authorization: Bearer sk_memopt_CNImxLNzKvHGiH9YDD4NP6u_lkyxLtnb" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "model": "meta-llama/Llama-2-7b-hf",
    "max_tokens": 100,
    "optimization_level": 7
  }'
```

## Deployment Steps

### 1. Deploy Worker (GPU Server)

```bash
cd /opt
git clone <repo> memopt
cd memopt/worker

# Generate and save token
export WORKER_TOKEN=$(openssl rand -hex 32)
echo $WORKER_TOKEN > ~/worker-token.txt

# Configure
cat > .env << ENVEOF
WORKER_TOKEN=$WORKER_TOKEN
MODEL_NAME=meta-llama/Llama-2-7b-hf
OPTIMIZATION_LEVEL=7
ENVEOF

# Enable GPU in docker-compose.yml (uncomment deploy section)

# Start
docker-compose up -d --build

# Verify
curl http://localhost:8001/health

# Configure firewall
sudo ufw allow from 72.60.26.210 to any port 8001
sudo ufw enable
```

### 2. Configure Control Plane (Coolify)

1. Add environment variables:
   - `WORKER_URL=http://GPU_SERVER_IP:8001`
   - `WORKER_TOKEN=<from worker-token.txt>`

2. Redeploy

3. Verify:
   ```bash
   curl https://memopt.sophisticatesai.com/health
   # Should show "worker": "healthy"
   ```

### 3. Test End-to-End

```bash
curl -X POST https://memopt.sophisticatesai.com/v1/infer \
  -H "Authorization: Bearer sk_memopt_CNImxLNzKvHGiH9YDD4NP6u_lkyxLtnb" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Test","model":"meta-llama/Llama-2-7b-hf","max_tokens":50,"optimization_level":7}'
```

## Troubleshooting Quick Reference

| Issue | Check | Solution |
|-------|-------|----------|
| Worker timeout | `docker-compose logs` | Increase `WORKER_TIMEOUT_SECONDS` |
| Connection refused | `telnet GPU_IP 8001` | Check firewall, worker running |
| Invalid token | Compare tokens | Ensure exact match |
| Model load error | `docker-compose logs` | Check disk space, internet |
| GPU not detected | `docker-compose logs` | Uncomment GPU section |

## Production Checklist

- [ ] Worker token generated (32+ chars)
- [ ] Worker started and healthy
- [ ] Firewall configured (only Coolify IP)
- [ ] Model loaded successfully
- [ ] Control plane env vars set
- [ ] Control plane redeployed
- [ ] Health check shows worker: healthy
- [ ] End-to-end test passes
- [ ] Monitoring configured

## Benefits

1. **Security**: Customer never accesses worker directly
2. **Isolation**: Control plane doesn't need GPU
3. **Scalability**: Add more workers independently
4. **IP Protection**: Inference code stays on your servers
5. **Flexibility**: Worker can be on any network (via Tailscale)
6. **Reliability**: Worker failures don't crash control plane
7. **Monitoring**: Separate health checks for each tier

## Next Steps

1. Deploy to production following DEPLOYMENT.md
2. Set up monitoring (Prometheus/Grafana)
3. Configure alerts for worker downtime
4. Test failover scenarios
5. Document runbooks for common issues
