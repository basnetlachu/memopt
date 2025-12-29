# MemOpt Two-Tier Production Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Customer Application                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Control Plane (Coolify VPS)                     │
│         https://memopt.sophisticatesai.com:3000             │
│                                                              │
│  - Authentication (API keys)                                 │
│  - Rate limiting & quotas                                    │
│  - Usage tracking & billing                                  │
│  - Request/response logging                                  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP + X-Worker-Token
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Data Plane (GPU Server)                         │
│                  Private Network :8001                       │
│                                                              │
│  - Model loading & caching                                   │
│  - GPU inference (OptimizedLLM)                             │
│  - Memory optimization                                       │
│  - Token authentication                                      │
└─────────────────────────────────────────────────────────────┘
```

## Security Model

- **Customer** → calls public API with `Authorization: Bearer <API_KEY>`
- **Control Plane** → validates customer, enforces quotas
- **Control Plane** → calls Worker with `X-Worker-Token: <WORKER_TOKEN>`
- **Worker** → rejects requests without valid token (401)
- **Customer** → NEVER has direct access to worker

## Part 1: Worker Deployment (GPU Server)

### Prerequisites

- GPU server with NVIDIA GPU (or CPU for slower inference)
- Docker and Docker Compose installed
- Network access FROM Coolify VPS (IP: 72.60.26.210)

### Step 1: Install NVIDIA Container Toolkit (GPU servers only)

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Step 2: Clone Repository

```bash
cd /opt
sudo git clone https://github.com/yourusername/memopt.git
cd memopt/worker
```

### Step 3: Configure Environment

```bash
# Generate secure token (32+ characters)
export WORKER_TOKEN=$(openssl rand -hex 32)
echo "Generated token: $WORKER_TOKEN"
# SAVE THIS TOKEN - you'll need it for Coolify

# Create .env file
cat > .env << EOF
WORKER_TOKEN=$WORKER_TOKEN
MODEL_NAME=meta-llama/Llama-2-7b-hf
OPTIMIZATION_LEVEL=7
WORKER_PORT=8001
# HF_TOKEN=your_huggingface_token  # Uncomment if using gated models
EOF
```

### Step 4: Enable GPU Support (if available)

Edit `docker-compose.yml` and uncomment the GPU section:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Step 5: Start Worker

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Wait for model to load (may take 5-10 minutes)
# Look for: "Model loaded successfully"
```

### Step 6: Verify Worker

```bash
# Check health (no auth required)
curl http://localhost:8001/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_name": "meta-llama/Llama-2-7b-hf",
#   "gpu_available": true,
#   "gpu_count": 1,
#   "gpu_name": "NVIDIA RTX 4090",
#   "optimization_level": 7
# }

# Test inference (requires token)
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

### Step 7: Configure Firewall

**CRITICAL: Only allow access from Coolify VPS**

```bash
# Allow only from Coolify IP
sudo ufw allow from 72.60.26.210 to any port 8001

# Enable firewall
sudo ufw enable

# Verify
sudo ufw status
```

**Alternative: Use Tailscale for private networking (recommended)**

```bash
# Install Tailscale on both servers
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Get Tailscale IP
tailscale ip -4
# Example: 100.64.1.2

# Use this IP for WORKER_URL in Coolify
```

## Part 2: Control Plane Configuration (Coolify)

### Step 1: Update Environment Variables

In Coolify dashboard for memopt service:

```bash
# Required in production
WORKER_URL=http://YOUR_GPU_SERVER_IP:8001
# If using Tailscale: http://100.64.1.2:8001

WORKER_TOKEN=<SAME_TOKEN_FROM_WORKER>

# Optional
WORKER_TIMEOUT_SECONDS=120
```

### Step 2: Redeploy Control Plane

1. Click "Redeploy" in Coolify
2. Check logs for startup validation
3. Look for successful startup (no errors about WORKER_URL or WORKER_TOKEN)

### Step 3: Verify Control Plane

```bash
# Check health
curl https://memopt.sophisticatesai.com/health

# Expected response:
# {
#   "status": "healthy",
#   "database": "healthy",
#   "redis": "disabled",
#   "worker": "healthy",  # <- Should be healthy
#   "version": "1.0.0",
#   "env": "production"
# }
```

## Part 3: End-to-End Testing

### Test 1: Health Checks

```bash
# Control plane health
curl https://memopt.sophisticatesai.com/health

# Worker health (from Coolify VPS only)
curl http://YOUR_GPU_SERVER_IP:8001/health
```

### Test 2: Full Inference Flow

```bash
# Use existing API key
API_KEY="sk_memopt_CNImxLNzKvHGiH9YDD4NP6u_lkyxLtnb"

# Test inference
curl -X POST https://memopt.sophisticatesai.com/v1/infer \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "model": "meta-llama/Llama-2-7b-hf",
    "max_tokens": 100,
    "temperature": 0.7,
    "optimization_level": 7
  }'

# Expected response:
# {
#   "generated_text": "Quantum computing is...",
#   "prompt_tokens": 6,
#   "completion_tokens": 94,
#   "total_tokens": 100
# }
```

### Test 3: Check Usage Tracking

```bash
# Get usage stats
curl https://memopt.sophisticatesai.com/v1/usage \
  -H "Authorization: Bearer $API_KEY"
```

## File Tree

```
memopt/
├── worker/
│   ├── main.py                 # Worker service
│   ├── requirements.txt        # Worker dependencies
│   ├── Dockerfile              # Worker container
│   ├── docker-compose.yml      # Worker deployment
│   └── .env.example            # Worker config template
├── saas/
│   ├── main.py                 # Control plane API (updated)
│   ├── config.py               # Settings (updated)
│   └── requirements.txt        # API dependencies (httpx added)
├── Dockerfile                  # Control plane container
└── DEPLOYMENT.md              # This file
```

## Troubleshooting

### Issue: Worker timeout

**Symptoms:**
```
"detail": "Worker service unavailable"
```

**Diagnosis:**
```bash
# From Coolify VPS, test worker connectivity
curl -v http://YOUR_GPU_SERVER_IP:8001/health

# Check firewall
sudo ufw status

# Check worker logs
docker-compose logs -f
```

**Solutions:**
- Increase `WORKER_TIMEOUT_SECONDS` in Coolify (default: 120)
- Check firewall allows Coolify IP
- Verify worker is running: `docker ps`
- Check worker health: `curl http://localhost:8001/health`

### Issue: Model download errors

**Symptoms:**
```
"model_load_error": "Connection timeout downloading model"
```

**Diagnosis:**
```bash
# Check internet connectivity
ping huggingface.co

# Check disk space
df -h

# View worker logs
docker-compose logs
```

**Solutions:**
- Ensure internet access from GPU server
- Check disk space (models are 5-50GB)
- For gated models, add `HF_TOKEN` to worker/.env
- Pre-download model:
  ```bash
  docker exec -it memopt-worker python -c "
  from transformers import AutoTokenizer, AutoModelForCausalLM
  AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
  AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
  "
  ```

### Issue: Invalid worker token

**Symptoms:**
```
{
  "detail": "Invalid worker token"
}
```

**Diagnosis:**
```bash
# Verify tokens match
# On GPU server:
cat worker/.env | grep WORKER_TOKEN

# In Coolify:
# Check WORKER_TOKEN environment variable
```

**Solutions:**
- Ensure WORKER_TOKEN matches exactly in both places
- Regenerate token and update both:
  ```bash
  openssl rand -hex 32
  ```

### Issue: Firewall blocked

**Symptoms:**
```
Connection refused or timeout from control plane
```

**Diagnosis:**
```bash
# From Coolify VPS (72.60.26.210):
telnet YOUR_GPU_SERVER_IP 8001
nc -zv YOUR_GPU_SERVER_IP 8001

# On GPU server:
sudo ufw status
sudo netstat -tlnp | grep 8001
```

**Solutions:**
```bash
# Allow Coolify IP
sudo ufw allow from 72.60.26.210 to any port 8001

# Or use Tailscale for private networking
```

### Issue: GPU not detected

**Symptoms:**
```
"gpu_available": false
```

**Diagnosis:**
```bash
# Check nvidia-smi works
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Solutions:**
- Uncomment GPU section in docker-compose.yml
- Install NVIDIA Container Toolkit
- Restart Docker: `sudo systemctl restart docker`
- CPU mode works but is slower

### Issue: Out of memory

**Symptoms:**
```
"CUDA out of memory" or worker crashes
```

**Solutions:**
- Use higher `OPTIMIZATION_LEVEL` (8-10)
- Use smaller model
- Add more GPU RAM
- Monitor: `nvidia-smi -l 1`

## Production Checklist

### Worker Server
- [ ] NVIDIA drivers installed (GPU only)
- [ ] Docker + nvidia-container-toolkit installed
- [ ] Worker token generated (32+ characters)
- [ ] Model name configured
- [ ] Firewall configured (only Coolify IP)
- [ ] Worker health check passes
- [ ] Test inference succeeds
- [ ] Logs show model loaded successfully

### Control Plane (Coolify)
- [ ] WORKER_URL configured
- [ ] WORKER_TOKEN configured (matches worker)
- [ ] Deployment successful
- [ ] Health check shows worker: healthy
- [ ] End-to-end inference test passes
- [ ] Usage tracking works

### Monitoring
- [ ] Set up worker monitoring (Prometheus/Grafana)
- [ ] Alert on worker downtime
- [ ] Monitor GPU usage: `nvidia-smi`
- [ ] Monitor worker logs: `docker-compose logs -f`
- [ ] Monitor API logs in Coolify

## Scaling

### Multiple GPUs (Same Server)

Run multiple worker instances on different GPUs:

```bash
# GPU 0
WORKER_PORT=8001 docker-compose up -d

# GPU 1 (edit compose to use device=1)
WORKER_PORT=8002 docker-compose up -d
```

Then add load balancing in control plane or use nginx.

### Multiple GPU Servers

Deploy worker to multiple servers, use comma-separated WORKER_URL or implement load balancing.

## Security Notes

1. **Worker Token**: 32+ characters, cryptographically random
2. **Firewall**: ONLY allow Coolify VPS IP
3. **Private Network**: Use Tailscale/Wireguard for additional security
4. **No Customer Access**: Worker is internal-only
5. **API Keys**: Never forwarded to worker
6. **Request ID**: Propagated for tracing

## Support

For issues:
1. Check worker logs: `docker-compose logs -f`
2. Check API logs in Coolify
3. Review troubleshooting section above
4. Verify all checklist items completed
