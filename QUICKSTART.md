# MemOpt Quick Start Guide

## TL;DR - Complete Deployment in 10 Minutes

### On GPU Server

```bash
# 1. Install prerequisites (one-time)
# See DEPLOYMENT.md for full NVIDIA toolkit installation

# 2. Clone and configure
cd /opt
sudo git clone <your-repo> memopt
cd memopt/worker

# 3. Generate token and create config
export WORKER_TOKEN=$(openssl rand -hex 32)
echo "Save this token: $WORKER_TOKEN"

cat > .env << 'ENVEOF'
WORKER_TOKEN=$WORKER_TOKEN
MODEL_NAME=meta-llama/Llama-2-7b-hf
OPTIMIZATION_LEVEL=7
ENVEOF

# 4. Enable GPU in docker-compose.yml (uncomment deploy section)

# 5. Start worker
docker-compose up -d --build

# 6. Wait for model to load (check logs)
docker-compose logs -f

# 7. Verify
curl http://localhost:8001/health

# 8. Configure firewall
sudo ufw allow from 72.60.26.210 to any port 8001
sudo ufw enable
```

### On Coolify

1. Add environment variables:
   - `WORKER_URL=http://YOUR_GPU_SERVER_IP:8001`
   - `WORKER_TOKEN=<token from above>`
2. Redeploy
3. Test: `curl https://memopt.sophisticatesai.com/health`

### Test End-to-End

```bash
curl -X POST https://memopt.sophisticatesai.com/v1/infer \
  -H "Authorization: Bearer sk_memopt_CNImxLNzKvHGiH9YDD4NP6u_lkyxLtnb" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "model": "meta-llama/Llama-2-7b-hf",
    "max_tokens": 50,
    "optimization_level": 7
  }'
```

## Key Commands

### Worker
```bash
docker-compose up -d      # Start
docker-compose down       # Stop
docker-compose logs -f    # Logs
curl http://localhost:8001/health  # Health
```

### Control Plane
```bash
curl https://memopt.sophisticatesai.com/health  # Health
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for full guide.
