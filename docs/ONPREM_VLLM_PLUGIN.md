# MemOpt On-Premise vLLM Plugin

## Overview

MemOpt is an on-premise inference engine plugin that optimizes vLLM deployments by reducing GPU memory bandwidth usage by 40-60% through:

- **INT8 KV Cache Quantization**: 4x memory reduction with <1% accuracy loss
- **Page-based Memory Management**: 40% reduction in fragmentation
- **Memory-aware Scheduling**: Dynamic batching with intelligent request routing
- **Layer-aware Retention**: Optimized cache management across transformer layers

MemOpt runs entirely within your infrastructure - no external API calls, no cloud dependencies.

## Architecture

MemOpt integrates with vLLM through runtime monkey-patching:

```
┌─────────────────────────────────────────┐
│  Your vLLM Deployment                   │
│  python -m vllm.entrypoints.openai...   │
└──────────────┬──────────────────────────┘
               │
               │ imports vLLM
               ▼
┌─────────────────────────────────────────┐
│  MemOpt Auto-Init (MEMOPT_ENABLED=1)    │
│  • Validates license (Ed25519)          │
│  • Patches vLLM components              │
│  • Enables optimizations                │
└──────────────┬──────────────────────────┘
               │
               │ patches applied
               ▼
┌─────────────────────────────────────────┐
│  vLLM Components (Optimized)            │
│  • Scheduler → Memory-aware batching    │
│  • KV Cache → INT8 quantization         │
│  • Memory Planner → Smart allocation    │
└─────────────────────────────────────────┘
```

**Key Design Principles:**

1. **Safe Degradation**: If patching fails, vLLM runs normally (unless `MEMOPT_STRICT=1`)
2. **Zero External Dependencies**: All validation happens offline with embedded public key
3. **Minimal Performance Overhead**: Patches add <1% latency while saving 40-60% bandwidth

## Installation

### 1. Install MemOpt

```bash
pip install memopt
```

Or from source:

```bash
git clone https://github.com/Memopt/Memopt.git
cd Memopt
pip install -e .
```

### 2. Install vLLM (if not already installed)

```bash
pip install vllm
```

### 3. Obtain License

Contact MemOpt sales to receive your `license.json` file. Place it in `/etc/memopt/`:

```bash
sudo mkdir -p /etc/memopt
sudo cp license.json /etc/memopt/
sudo chmod 644 /etc/memopt/license.json
```

**License Format:**

```json
{
  "customer_id": "your-company-id",
  "tier": "enterprise",
  "issued_at": "2025-01-01T00:00:00Z",
  "expires_at": "2026-01-01T00:00:00Z",
  "max_gpus": 8,
  "features": ["vllm", "quantization", "profiling"],
  "telemetry_required": false,
  "signature": "base64-encoded-ed25519-signature"
}
```

The license is cryptographically signed with Ed25519. MemOpt validates the signature using an embedded public key - no internet connection required.

## Configuration

MemOpt uses environment variables for configuration:

### Required Variables

```bash
export MEMOPT_ENABLED=1                           # Enable MemOpt plugin
export MEMOPT_LICENSE_PATH=/etc/memopt/license.json  # License location
```

### Optional Variables

```bash
export MEMOPT_SAFE_MODE=1     # Only static optimizations (no scheduler patching)
export MEMOPT_STRICT=1        # Fail if patches don't apply (default: graceful degradation)
export MEMOPT_DEV_MODE=1      # Bypass license validation for development
```

**Safe Mode (`MEMOPT_SAFE_MODE=1`):**

Enables only static optimizations (KV cache configuration) and disables dynamic patches (scheduler hooks). Use this for:
- First production deployment
- Maximum performance safety
- Validating compatibility with new vLLM versions

**Expected behavior:**
- Memory optimization: ✅ (INT8 quantization, page-based allocation)
- Scheduler optimization: ❌ (disabled for safety)
- Performance impact: Zero (guaranteed no regression)

## Usage

### Basic Usage with vLLM

Set environment variables and run vLLM normally:

```bash
export MEMOPT_ENABLED=1
export MEMOPT_LICENSE_PATH=/etc/memopt/license.json

python -m vllm.entrypoints.openai.api_server \
  --model /models/llama-2-13b \
  --host 0.0.0.0 \
  --port 8000
```

MemOpt will automatically:

1. Validate your license on startup
2. Detect vLLM version
3. Apply optimization patches
4. Print patch status to console

**Expected Output:**

```
[MemOpt] License validated for customer: your-company-id
[MemOpt] Detected vLLM version: 0.3.2
[MemOpt] Applying vLLM patches...
[MemOpt] Successfully patched: scheduler, kv_cache, memory_planner
[MemOpt] vLLM plugin initialized successfully
INFO:     Started server process [12345]
```

### With Docker

Add environment variables to your Dockerfile or docker-compose:

**docker-compose.yml:**

```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    environment:
      - MEMOPT_ENABLED=1
      - MEMOPT_LICENSE_PATH=/etc/memopt/license.json
    volumes:
      - ./license.json:/etc/memopt/license.json:ro
      - ./models:/models:ro
    command: >
      --model /models/llama-2-13b
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Kubernetes Deployment

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-memopt
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
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
        - name: models
          mountPath: /models
          readOnly: true
        resources:
          limits:
            nvidia.com/gpu: 2
      volumes:
      - name: license
        secret:
          secretName: memopt-license
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

**Create license secret:**

```bash
kubectl create secret generic memopt-license \
  --from-file=license.json=/etc/memopt/license.json
```

## Diagnostics

MemOpt provides a `doctor` command to verify your installation:

```bash
memopt doctor
```

**Sample Output:**

```
MemOpt System Diagnostics

============================================================
License Status
============================================================
License path: /etc/memopt/license.json
Customer ID: acme-corp
Tier: enterprise
Expires: 2026-01-01T00:00:00Z
Max GPUs: 8
Features: vllm, quantization, profiling
Telemetry: Optional
Status: VALID

============================================================
vLLM Status
============================================================
Installed: Yes
Version: 0.3.2
Location: /usr/local/lib/python3.10/site-packages/vllm/__init__.py

============================================================
GPU Status
============================================================
CUDA: Available
CUDA Version: 12.1
GPU Count: 2
  GPU 0: NVIDIA A100-SXM4-40GB
    Memory: 40.00 GB
    Compute: 8.0
  GPU 1: NVIDIA A100-SXM4-40GB
    Memory: 40.00 GB
    Compute: 8.0

============================================================
MemOpt Plugin Status
============================================================
MEMOPT_ENABLED: True
MEMOPT_STRICT: False
Status: ACTIVE
vLLM Version: 0.3.2
Patches Applied:
  - scheduler: OK
  - kv_cache: OK
  - memory_planner: OK

============================================================
Summary
============================================================
Status: ALL SYSTEMS OPERATIONAL

MemOpt is ready to use!
Run vLLM with MEMOPT_ENABLED=1 to activate optimizations.
```

## Troubleshooting

### License Validation Failed

**Error:**

```
License validation failed: Invalid signature
```

**Solutions:**

1. Verify license file integrity - it may be corrupted
2. Contact MemOpt support for a new license
3. Use dev mode for testing: `export MEMOPT_DEV_MODE=1`

### License Expired

**Error:**

```
License expired on 2025-12-31T23:59:59Z
```

**Solution:**

Contact MemOpt sales to renew your license.

### vLLM Not Detected

**Warning:**

```
vLLM not detected. MemOpt vLLM plugin disabled.
Install vLLM to use MemOpt optimizations.
```

**Solution:**

```bash
pip install vllm
```

### Patch Failed

**Warning:**

```
[MemOpt] Failed to patch some components:
  - scheduler: Import error: No module named 'vllm.core.scheduler'
```

**Explanation:**

vLLM's internal structure changed between versions. MemOpt tries multiple fallback strategies.

**Solutions:**

1. **Safe Degradation (Default)**: vLLM will run normally without MemOpt optimizations
2. **Strict Mode**: Set `MEMOPT_STRICT=1` to fail startup if patches don't apply
3. **Update MemOpt**: Check for newer version compatible with your vLLM version
4. **Contact Support**: Report the vLLM version for compatibility update

### OOM Errors After Installing MemOpt

**Issue:**

Getting out-of-memory errors after enabling MemOpt.

**Diagnosis:**

This should not happen - MemOpt reduces memory usage. If you see OOM:

1. Run `memopt doctor` to verify patch status
2. Check vLLM logs for memory allocation messages
3. Disable MemOpt temporarily: `export MEMOPT_ENABLED=0`

**If OOM persists without MemOpt:**

This is a vLLM configuration issue, not MemOpt-related. Adjust vLLM's `--gpu-memory-utilization` flag.

## Performance Expectations

### Memory Savings

- **KV Cache**: 4x reduction via INT8 quantization
- **Fragmentation**: 40% reduction via page-based allocation
- **Overall**: 40-60% total memory bandwidth reduction

### Throughput Impact

- **Best Case**: +20-30% throughput (memory-bound workloads)
- **Typical**: Neutral to +10% throughput
- **Worst Case**: -1-2% throughput (compute-bound workloads)

### Latency Impact

- **Per-token Latency**: <1% increase (quantization overhead)
- **Time to First Token**: No change
- **End-to-End**: Typically faster due to better batching

## Advanced Configuration

### Environment-Specific Overrides

For different environments (dev/staging/prod), use separate license files:

```bash
# Development
export MEMOPT_LICENSE_PATH=/etc/memopt/license-dev.json

# Staging
export MEMOPT_LICENSE_PATH=/etc/memopt/license-staging.json

# Production
export MEMOPT_LICENSE_PATH=/etc/memopt/license-prod.json
```

### Monitoring Integration

MemOpt logs to stdout/stderr. Integrate with your monitoring stack:

**Prometheus Example (parse logs):**

```bash
vllm_memopt_patches_applied{component="scheduler"} 1
vllm_memopt_patches_applied{component="kv_cache"} 1
vllm_memopt_memory_saved_gb 24.5
```

**CloudWatch Example:**

Forward logs to CloudWatch and create metric filters for:

- `[MemOpt] Successfully patched:` → Success metric
- `[MemOpt] Failed to patch` → Alert trigger

## Security Considerations

### License File Protection

The license file contains your cryptographic signature. Protect it:

```bash
# Set restrictive permissions
sudo chown root:root /etc/memopt/license.json
sudo chmod 644 /etc/memopt/license.json
```

In Kubernetes, use Secrets with restrictive RBAC:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: memopt-license
type: Opaque
data:
  license.json: <base64-encoded-license>
```

### No Telemetry by Default

MemOpt does **not** send telemetry unless `telemetry_required: true` in your license. Even then, only aggregate metrics (no prompts/completions) are sent.

To explicitly disable telemetry:

```bash
export MEMOPT_TELEMETRY=0
```

### Code Signing Verification

The MemOpt wheel is signed. Verify before installation:

```bash
pip install memopt --require-hashes
```

(Contact support for official hash manifests)

## Support

### Community Support

- GitHub Issues: https://github.com/Memopt/Memopt/issues
- Documentation: https://docs.memopt.ai

### Enterprise Support

Enterprise customers receive:

- Dedicated Slack channel
- 24/7 on-call support
- Custom vLLM version compatibility
- Integration assistance

Contact: support@memopt.ai

## License Tiers

### Enterprise Flat ($800k/year)

- Up to 16 GPUs
- All features (vLLM, quantization, profiling)
- Enterprise support
- No telemetry required

### Revenue Share (35% of savings)

- Unlimited GPUs
- All features
- Standard support
- Monthly telemetry required for billing

## Frequently Asked Questions

### Q: Does MemOpt modify vLLM's code on disk?

**A:** No. MemOpt applies runtime monkey-patches to vLLM's in-memory objects. The original vLLM installation is never modified.

### Q: What happens if my license expires?

**A:** MemOpt will refuse to initialize. vLLM will run normally without optimizations. No downtime.

### Q: Can I use MemOpt without vLLM?

**A:** Yes. MemOpt includes standalone inference capabilities via `OptimizedLLM` class. However, the primary use case is the vLLM plugin.

### Q: Is MemOpt compatible with vLLM's OpenAI-compatible API?

**A:** Yes. MemOpt is transparent to clients - all vLLM APIs work identically.

### Q: What vLLM versions are supported?

**A:** MemOpt supports vLLM 0.2.0+. Tested on 0.2.x, 0.3.x, 0.4.x. Use `memopt doctor` to verify compatibility.

### Q: Can I disable specific optimizations?

**A:** Not currently. MemOpt applies all available patches. This is intentional for simplicity. If specific patches fail, MemOpt degrades gracefully.

### Q: Does MemOpt work with multi-GPU setups?

**A:** Yes. MemOpt optimizes memory on each GPU independently. Ensure your license covers the GPU count.

---

**MemOpt Version:** 0.1.0
**Last Updated:** 2025-12-30
**License:** Proprietary
