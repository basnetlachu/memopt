# memopt Deployment Guide

## You received:
- License key: `sk-lic-xxxx` (in your welcome email)
- This deployment package

## Choose your deployment:

### Option A: Single Server (simplest)
One GPU server. Full memopt stack on one machine.
→ See [single-server/](single-server/)

### Option B: Multi-Server Kubernetes
Many GPU servers. Centralized management.
→ See [kubernetes/](kubernetes/)

---

## Option A: Single Server

**Requirements:**
- Ubuntu 20.04+ or similar Linux
- NVIDIA drivers installed (`nvidia-smi` returns output)
- Docker + nvidia-container-toolkit
- 8GB+ system RAM

### Step 1: Install Docker (if needed)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2: Install nvidia-container-toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 3: Configure memopt

```bash
cd deployment/single-server/
cp .env.example .env
nano .env   # Add your MEMOPT_LICENSE_KEY and NODE_NAME
```

### Step 4: Start memopt

```bash
docker-compose up -d
```

### Step 5: Verify

```bash
# Health check (no auth required)
curl http://localhost:8080/health

# View dashboard
open http://localhost:8080
```

### Step 6: Run your first scan

```bash
docker exec memopt memopt scan
```

That's it. memopt is now monitoring all GPUs on this server.
Auto-optimization starts within 60 seconds.

---

## Option B: Multi-Server Kubernetes

**Requirements:**
- Kubernetes cluster with GPU nodes
- `kubectl` configured with cluster access
- Helm 3.x installed
- NVIDIA device plugin installed on cluster nodes

### Step 1: Add memopt Helm repo

```bash
helm repo add memopt https://charts.memopt.ai \
  --username your-email@company.com \
  --password sk-lic-xxxx
helm repo update
```

### Step 2: Install

```bash
helm install memopt memopt/memopt \
  --namespace memopt \
  --create-namespace \
  --set license.key=sk-lic-xxxx \
  --set gpu.limit=0 \
  --set daemon.gpuCostPerHour=8.00 \
  --set daemon.autoApply=false
```

### Step 3: Access dashboard

```bash
kubectl port-forward -n memopt svc/memopt-control-plane 8080:8080
open http://localhost:8080
```

### Step 4: View the API key (set on other nodes)

```bash
kubectl logs -n memopt deployment/memopt-control-plane | grep "API Key"
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `MEMOPT_LICENSE_KEY` | **Yes** | — | Your license key from welcome email |
| `NODE_NAME` | **Yes** | hostname | Unique name for this server (appears in dashboard) |
| `MEMOPT_GPU_COST_PER_HOUR` | No | `2.50` | Your actual GPU cost in USD/hour (for ROI reports) |
| `MEMOPT_AUTO_APPLY` | No | `false` | Auto-apply optimizations (start with `false`) |
| `MEMOPT_SCAN_INTERVAL` | No | `60` | Seconds between GPU scans |
| `MEMOPT_API_KEY` | No | auto-generated | API key for dashboard — set after first run to persist |
| `MEMOPT_CONTROL_PLANE` | No | — | URL of control plane server (multi-node setup) |
| `MEMOPT_HTTPS` | No | `false` | Enable TLS termination via nginx |
| `MEMOPT_DOMAIN` | No | `localhost` | Domain name (needed for TLS cert) |
| `MEMOPT_MODE` | No | `full` | `full` \| `daemon` \| `control-plane` \| `scan` |

## GPU Cost Reference

| Hardware | Cloud Provider | Cost/hr |
|---|---|---|
| H100 SXM | AWS p5, Lambda, CoreWeave | ~$8.00 |
| A100 80GB | AWS p4, Azure NDv4 | ~$3.00 |
| A100 40GB | CoreWeave, Lambda | ~$2.00 |
| RTX 4090 | Vast.ai, Lambda | ~$0.75 |
| On-prem | Calculate: hardware ÷ (useful_life_hours) | varies |

## Security Notes

- The API key is auto-generated on first start and saved to `~/.memopt/api_key` (chmod 600)
- Copy the API key from the logs after first run: set it as `MEMOPT_API_KEY` in `.env` to persist across restarts
- `/health` endpoint never requires authentication — required for Kubernetes probes
- For production: set `MEMOPT_HTTPS=true` and provide `MEMOPT_DOMAIN`

## Support

- Email: support@memopt.ai
- Response time: 4 hours (Enterprise), 24 hours (Pro)
- Docs: https://docs.memopt.ai
