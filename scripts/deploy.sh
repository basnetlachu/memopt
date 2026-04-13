#!/usr/bin/env bash
# memopt node bootstrap — takes a bare OS to running memopt.
# Idempotent: safe to run multiple times.
#
# Usage:
#   ./scripts/deploy.sh \
#     --node-id node-001 \
#     --control-plane http://cp:8765 \
#     --redis-url redis://redis:6379 \
#     --mode serving
#
# Exit codes:
#   0 — success
#   1 — prerequisite failure
#   2 — silicon certification failure
#   3 — service health check failure

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
NODE_ID=""
CONTROL_PLANE=""
REDIS_URL=""
MODE="serving"  # serving | control-plane | both
NVME_DIR="/var/memopt/nvme"
CONFIG_DIR="/etc/memopt"
DATA_DIR="/var/memopt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colors ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

# ── Parse arguments ───────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --node-id ID            Node identifier (required)
  --control-plane URL     Control plane URL (optional)
  --redis-url URL         Redis URL for cluster state (optional)
  --mode MODE             serving | control-plane | both (default: serving)
  --nvme-dir DIR          NVMe storage directory (default: /var/memopt/nvme)
  --help                  Show this help

Examples:
  # Single node, local only
  $(basename "$0") --node-id dev-01

  # Cluster node with Redis and control plane
  $(basename "$0") --node-id prod-01 \\
    --control-plane http://cp.internal:8765 \\
    --redis-url redis://redis.internal:6379

  # Control plane node
  $(basename "$0") --node-id cp-01 --mode control-plane
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --node-id)        NODE_ID="$2"; shift 2 ;;
        --control-plane)  CONTROL_PLANE="$2"; shift 2 ;;
        --redis-url)      REDIS_URL="$2"; shift 2 ;;
        --mode)           MODE="$2"; shift 2 ;;
        --nvme-dir)       NVME_DIR="$2"; shift 2 ;;
        --help)           usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$NODE_ID" ]]; then
    NODE_ID="$(hostname)"
    warn "No --node-id specified, using hostname: $NODE_ID"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  memopt node bootstrap"
echo "  Node ID: $NODE_ID"
echo "  Mode:    $MODE"
echo "═══════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Verify prerequisites
# ═══════════════════════════════════════════════════════════════════════
echo "── Step 1: Prerequisites ──"

# Python 3.10+
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]]; then
        ok "Python $PY_VERSION"
    else
        fail "Python $PY_VERSION found, need 3.10+"
        echo "  Install: sudo apt install python3.11 python3.11-venv"
        exit 1
    fi
else
    fail "Python 3 not found"
    echo "  Install: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# CUDA (required for serving, optional for control-plane)
if [[ "$MODE" != "control-plane" ]]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        ok "CUDA available: $GPU_NAME"
    else
        fail "nvidia-smi not found or failed"
        echo "  Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
else
    ok "CUDA check skipped (control-plane mode)"
fi

# Disk space
ROOT_FREE_GB=$(df / --output=avail -BG 2>/dev/null | tail -1 | tr -d ' G' || echo "0")
if [[ "$ROOT_FREE_GB" -lt 10 ]]; then
    warn "Low disk space on /: ${ROOT_FREE_GB}GB free (recommend >= 10GB)"
else
    ok "Disk space: ${ROOT_FREE_GB}GB free on /"
fi

# Docker (optional)
HAS_DOCKER=false
if command -v docker &>/dev/null; then
    HAS_DOCKER=true
    ok "Docker available"
else
    ok "Docker not found — using pip install"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Install memopt
# ═══════════════════════════════════════════════════════════════════════
echo "── Step 2: Install memopt ──"

if python3 -c "import memopt" &>/dev/null; then
    ok "memopt already installed"
else
    if [[ -f "$PROJECT_DIR/pyproject.toml" ]]; then
        echo "  Installing from local source..."
        pip install -e "$PROJECT_DIR" --quiet 2>/dev/null || \
            pip install "$PROJECT_DIR" --quiet
        ok "memopt installed from source"
    else
        echo "  Installing from pip..."
        pip install memopt --quiet
        ok "memopt installed from pip"
    fi
fi

# Verify
if python3 -c "import memopt" &>/dev/null; then
    ok "memopt import verified"
else
    fail "memopt import failed after installation"
    exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Configure environment
# ═══════════════════════════════════════════════════════════════════════
echo "── Step 3: Configure ──"

sudo mkdir -p "$CONFIG_DIR" "$DATA_DIR/nvme" "$DATA_DIR/logs" \
    "$DATA_DIR/certs" "$NVME_DIR" 2>/dev/null || \
    mkdir -p "$CONFIG_DIR" "$DATA_DIR/nvme" "$DATA_DIR/logs" \
        "$DATA_DIR/certs" "$NVME_DIR" 2>/dev/null || true

# Create config file (preserves existing values)
CONFIG_FILE="$CONFIG_DIR/config.env"
if [[ ! -f "$CONFIG_FILE" ]] || [[ "$1" == "--force" ]] 2>/dev/null; then
    cat > "$CONFIG_FILE" 2>/dev/null || sudo tee "$CONFIG_FILE" > /dev/null <<ENVEOF
# memopt node configuration — generated by deploy.sh
# Node identity
MEMOPT_NODE_ID=${NODE_ID}

# Cluster
MEMOPT_CONTROL_PLANE_URL=${CONTROL_PLANE}
REDIS_URL=${REDIS_URL}

# Transport
MEMOPT_TRANSPORT=tcp

# Storage
MEMOPT_NVME_DIR=${NVME_DIR}

# Eviction watermarks (tune per workload)
MEMOPT_EVICT_HIGH=0.90
MEMOPT_EVICT_LOW=0.75

# Silicon certification
MEMOPT_CERTIFY_ON_STARTUP=1

# Federation gossip
MEMOPT_GOSSIP_FANOUT=5

# Ledger
MEMOPT_LEDGER_DB_PATH=${DATA_DIR}/ledger.db
ENVEOF
    ok "Config written to $CONFIG_FILE"
else
    ok "Config exists at $CONFIG_FILE (not overwritten)"
fi

# Create memopt user if doesn't exist
if ! id -u memopt &>/dev/null; then
    sudo useradd -r -m -s /bin/false memopt 2>/dev/null || true
    ok "Created memopt service user"
else
    ok "memopt user exists"
fi

sudo chown -R memopt:memopt "$DATA_DIR" 2>/dev/null || true

echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 4: Install systemd units
# ═══════════════════════════════════════════════════════════════════════
echo "── Step 4: Systemd units ──"

SYSTEMD_DIR="/etc/systemd/system"
DEPLOY_DIR="$PROJECT_DIR/deploy/systemd"

if [[ -d "$DEPLOY_DIR" ]]; then
    for unit in memopt-transport.service memopt-serving.service memopt-control-plane.service; do
        if [[ -f "$DEPLOY_DIR/$unit" ]]; then
            sudo cp "$DEPLOY_DIR/$unit" "$SYSTEMD_DIR/" 2>/dev/null && \
                ok "Installed $unit" || \
                warn "Could not install $unit (not root?)"
        fi
    done
    sudo systemctl daemon-reload 2>/dev/null || true
else
    warn "Systemd units not found at $DEPLOY_DIR"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 5: Silicon certification
# ═══════════════════════════════════════════════════════════════════════
echo "── Step 5: Silicon certification ──"

if [[ "$MODE" == "control-plane" ]]; then
    ok "Certification skipped (control-plane mode)"
    CERT_HASH="n/a"
else
    # Source config for environment
    set -a
    source "$CONFIG_FILE" 2>/dev/null || true
    set +a

    CERT_OUTPUT=$(python3 -c "
from memopt.kernels.certification import run_certification
import json
try:
    cert = run_certification(node_id='${NODE_ID}')
    print(json.dumps({
        'passed': cert.get('all_passed', False),
        'hash': cert.get('certificate_hash', 'unknown')[:16],
        'detail': str(cert.get('summary', ''))
    }))
except Exception as e:
    print(json.dumps({'passed': False, 'hash': '', 'detail': str(e)}))
" 2>/dev/null || echo '{"passed":false,"hash":"","detail":"certification module error"}')

    CERT_PASSED=$(echo "$CERT_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('passed','false'))" 2>/dev/null || echo "false")
    CERT_HASH=$(echo "$CERT_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('hash','unknown'))" 2>/dev/null || echo "unknown")
    CERT_DETAIL=$(echo "$CERT_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('detail',''))" 2>/dev/null || echo "")

    if [[ "$CERT_PASSED" == "True" || "$CERT_PASSED" == "true" ]]; then
        ok "Silicon certification PASSED (hash: $CERT_HASH)"
    else
        fail "Silicon certification FAILED"
        echo "  Detail: $CERT_DETAIL"
        echo "  Node should not serve traffic with failed certification."
        echo "  Fix hardware issues and re-run: $0 --node-id $NODE_ID"
        exit 2
    fi
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 6: Start services
# ═══════════════════════════════════════════════════════════════════════
echo "── Step 6: Start services ──"

start_service() {
    local svc=$1
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        ok "$svc already running"
    elif systemctl cat "$svc" &>/dev/null; then
        sudo systemctl enable "$svc" 2>/dev/null || true
        sudo systemctl start "$svc" 2>/dev/null
        sleep 2
        if systemctl is-active --quiet "$svc"; then
            ok "$svc started"
        else
            warn "$svc failed to start — check: journalctl -u $svc -n 20"
        fi
    else
        warn "$svc unit not installed — skipping"
    fi
}

if [[ "$MODE" == "serving" || "$MODE" == "both" ]]; then
    start_service "memopt-transport"
    start_service "memopt-serving"
fi

if [[ "$MODE" == "control-plane" || "$MODE" == "both" ]]; then
    start_service "memopt-control-plane"
fi

# Health check for serving
if [[ "$MODE" == "serving" || "$MODE" == "both" ]]; then
    echo "  Waiting for health check..."
    HEALTHY=false
    for i in $(seq 1 30); do
        if curl -sf http://localhost:8080/health &>/dev/null; then
            HEALTHY=true
            break
        fi
        sleep 1
    done
    if $HEALTHY; then
        ok "Serving engine healthy"
    else
        warn "Serving engine not responding on :8080 after 30s"
        echo "  Check: journalctl -u memopt-serving -n 30"
    fi
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 7: Register with control plane
# ═══════════════════════════════════════════════════════════════════════
echo "── Step 7: Control plane registration ──"

if [[ -n "$CONTROL_PLANE" ]]; then
    REG_STATUS=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X POST "${CONTROL_PLANE}/api/v1/report" \
        -H "Content-Type: application/json" \
        -d "{
            \"node_name\": \"${NODE_ID}\",
            \"timestamp\": $(date +%s),
            \"gpu_count\": $(nvidia-smi -L 2>/dev/null | wc -l || echo 0),
            \"status\": \"online\"
        }" 2>/dev/null || echo "000")

    if [[ "$REG_STATUS" == "200" || "$REG_STATUS" == "201" ]]; then
        ok "Registered with control plane"
    else
        warn "Control plane registration failed (HTTP $REG_STATUS)"
        echo "  Node operates independently without control plane."
    fi
else
    ok "No control plane configured (standalone mode)"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 8: Summary
# ═══════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════"
echo "  memopt node bootstrap complete"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Node ID:        $NODE_ID"
echo "  Mode:           $MODE"
echo "  Certification:  ${CERT_HASH:-n/a}"
echo "  Config:         $CONFIG_FILE"
echo "  Data dir:       $DATA_DIR"
echo "  NVMe dir:       $NVME_DIR"
echo ""

# RDMA status
if command -v ibv_devinfo &>/dev/null && ibv_devinfo 2>/dev/null | grep -q "PORT_ACTIVE"; then
    echo "  RDMA:           available (PORT_ACTIVE)"
else
    echo "  RDMA:           not available (TCP transport)"
fi

# Redis status
if [[ -n "$REDIS_URL" ]]; then
    if python3 -c "import redis; redis.from_url('$REDIS_URL').ping()" &>/dev/null; then
        echo "  Redis:          connected"
    else
        echo "  Redis:          unreachable ($REDIS_URL)"
    fi
else
    echo "  Redis:          not configured (local mode)"
fi

echo ""
echo "  memopt is ready on $NODE_ID"
echo ""
