# MemOpt Deployment Guide

## Important: Architecture Change

**MemOpt is now an on-premise Python package, not a containerized service.**

### ❌ Old Model (Deprecated)
- SaaS deployment on Coolify
- Separate control plane + GPU worker containers
- Customers send requests to our servers

### ✅ New Model (Current)
- Customers install MemOpt package in their infrastructure
- Plugin integrates with their existing vLLM deployment
- Zero network calls, complete data privacy

## Production Deployment

### Customer Installation

Customers install MemOpt in their vLLM environment:

```bash
# 1. Install MemOpt
pip install memopt

# 2. Place license file
sudo mkdir -p /etc/memopt
sudo cp license.json /etc/memopt/

# 3. Configure environment
export MEMOPT_ENABLED=1
export MEMOPT_LICENSE_PATH=/etc/memopt/license.json

# 4. Run vLLM normally (MemOpt auto-attaches)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-hf \
  --port 8000
```

MemOpt automatically optimizes vLLM when imported.

### Docker Deployment (Customer-Side)

Customers can add MemOpt to their vLLM Docker images:

```dockerfile
FROM vllm/vllm-openai:latest

# Install MemOpt
RUN pip install memopt

# Copy license (from Docker build context)
COPY license.json /etc/memopt/license.json

# Environment variables
ENV MEMOPT_ENABLED=1
ENV MEMOPT_LICENSE_PATH=/etc/memopt/license.json

# vLLM will auto-initialize MemOpt
CMD ["--model", "/models/llama", "--port", "8000"]
```

### Kubernetes Deployment (Customer-Side)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-with-memopt
spec:
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        env:
        - name: MEMOPT_ENABLED
          value: "1"
        - name: MEMOPT_LICENSE_PATH
          value: /etc/memopt/license.json
        volumeMounts:
        - name: license
          mountPath: /etc/memopt
          readOnly: true
      volumes:
      - name: license
        secret:
          secretName: memopt-license
```

## Demo Container (This Repository)

The Dockerfile in this repository is for **demonstration/testing only**:

```bash
# Build demo container
docker build -t memopt-demo .

# Run demo
docker run -it memopt-demo

# Output shows:
# - MemOpt installed successfully
# - memopt doctor diagnostics
# - Installation instructions
```

This container demonstrates:
- ✅ MemOpt package installation
- ✅ CLI (`memopt doctor`) functionality
- ✅ License validation (dev mode)

This container does NOT:
- ❌ Run inference (no vLLM installed)
- ❌ Provide API endpoints (not a service)
- ❌ Replace customer vLLM deployment

## Distribution Model

### Package Distribution

**PyPI (Recommended):**
```bash
pip install memopt
```

**From Source:**
```bash
git clone https://github.com/Memopt/Memopt.git
cd Memopt
pip install -e .
```

### License Distribution

Send customers a signed `license.json` file:

```json
{
  "customer_id": "acme-corp",
  "tier": "enterprise",
  "expires_at": "2026-01-01T00:00:00Z",
  "max_gpus": 8,
  "features": ["vllm", "quantization", "profiling"],
  "signature": "base64-ed25519-signature"
}
```

License is validated **offline** (no phone-home).

## Sales Model

### Enterprise Flat ($800k/year)
- Customer receives: `license.json` file
- Customer installs: `pip install memopt`
- No ongoing service hosting required

### Revenue Share (35% of savings)
- Same installation process
- License includes `telemetry_required: true`
- MemOpt reports usage metrics for billing

## Support Model

### Self-Service
- Documentation: https://docs.memopt.ai
- GitHub Issues: https://github.com/Memopt/Memopt/issues

### Enterprise Support
- Slack channel
- Email: support@memopt.ai
- 24/7 on-call for critical issues

## Migration from Old SaaS Model

If you have existing SaaS deployments (Coolify):

1. **Notify customers** of architecture change
2. **Provide migration guide** (see docs/ONPREM_VLLM_PLUGIN.md)
3. **Send new licenses** (offline validation)
4. **Decommission SaaS infrastructure** (Coolify, GPU workers)

## Why This Change?

**Customer feedback:**
- ❌ Don't want prompts sent to external servers (privacy)
- ❌ Don't want network latency
- ❌ Don't want vendor lock-in
- ✅ Want to run in their own infrastructure
- ✅ Want complete data ownership

**Business benefits:**
- ✅ Lower operational costs (no infrastructure to maintain)
- ✅ Easier scaling (customers scale their own deployment)
- ✅ Better security story (offline license, no telemetry by default)
- ✅ Simpler support (just a Python package)

## Questions?

**Is the Coolify deployment still used?**
No. The Dockerfile in this repo is for demonstration only.

**How do customers get updates?**
`pip install --upgrade memopt`

**How is the license enforced?**
Ed25519 cryptographic signature, validated offline on startup.

**What if a customer's license expires?**
MemOpt refuses to initialize. vLLM runs normally without optimizations.

**Can customers use MemOpt without a license?**
Only in dev mode (`MEMOPT_DEV_MODE=1`) for testing. Production requires valid license.

---

**For production deployment documentation, see:**
- [docs/ONPREM_VLLM_PLUGIN.md](docs/ONPREM_VLLM_PLUGIN.md)
- [docs/PERFORMANCE_SAFETY.md](docs/PERFORMANCE_SAFETY.md)
