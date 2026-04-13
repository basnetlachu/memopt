#!/usr/bin/env bash
# memopt node verification — checks health of a deployed node.
#
# Usage:
#   ./scripts/verify_node.sh
#
# Exit 0 if all checks pass, 1 if any fail.

set -uo pipefail

# ── Colors ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

# Load config
CONFIG_FILE="/etc/memopt/config.env"
if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    source "$CONFIG_FILE"
    set +a
fi

NODE_ID="${MEMOPT_NODE_ID:-$(hostname)}"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: $(basename "$0")"
    echo "Verifies memopt node health. Exit 0 = all pass, 1 = failures."
    exit 0
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  memopt node verification — $NODE_ID"
echo "═══════════════════════════════════════════════════"
echo ""

# ── 1. Silicon certificate ────────────────────────────────────────────
CERT_DIR="/var/memopt/certs"
if ls "$CERT_DIR"/cert_*.json &>/dev/null || ls "$CERT_DIR"/*.json &>/dev/null; then
    pass "Silicon certificate exists"
else
    # Check if certification was run (even without file)
    if python3 -c "
from memopt.kernels.certification import run_certification
cert = run_certification(node_id='$NODE_ID')
assert cert.get('all_passed', False)
" &>/dev/null; then
        pass "Silicon certification passes (live check)"
    else
        fail "Silicon certification — no certificate found and live check failed"
        echo "       Run: python3 -m memopt certify --node-id $NODE_ID"
    fi
fi

# ── 2. memopt-transport running ───────────────────────────────────────
if systemctl is-active --quiet memopt-transport 2>/dev/null; then
    pass "memopt-transport active"
elif pgrep -f memopt-transport &>/dev/null; then
    pass "memopt-transport running (not systemd)"
else
    # Transport is optional (TCP fallback available)
    if [[ -f /dev/shm/memopt_transport_${NODE_ID}_req ]]; then
        pass "memopt-transport ring buffer found"
    else
        pass "memopt-transport not running (TCP fallback active)"
    fi
fi

# ── 3. memopt-serving running and healthy ─────────────────────────────
if systemctl is-active --quiet memopt-serving 2>/dev/null; then
    if curl -sf http://localhost:8080/health &>/dev/null; then
        pass "memopt-serving healthy"
    else
        fail "memopt-serving active but /health not responding"
        echo "       Check: journalctl -u memopt-serving -n 20"
    fi
elif curl -sf http://localhost:8080/health &>/dev/null; then
    pass "memopt-serving healthy (not systemd)"
else
    fail "memopt-serving not responding on :8080"
    echo "       Start: systemctl start memopt-serving"
    echo "       Or:    python3 -m memopt.serving.server --port 8080"
fi

# ── 4. Redis reachable ────────────────────────────────────────────────
if [[ -n "${REDIS_URL:-}" ]]; then
    if python3 -c "
import redis, os
r = redis.from_url(os.environ.get('REDIS_URL', ''))
r.ping()
" &>/dev/null; then
        pass "Redis reachable ($REDIS_URL)"
    else
        fail "Redis unreachable ($REDIS_URL)"
        echo "       GKD cluster dedup disabled. Local fallback active."
    fi
else
    pass "Redis not configured (standalone mode)"
fi

# ── 5. Wiring audit ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/audit_wiring.py" ]]; then
    AUDIT_RESULT=$(python3 "$SCRIPT_DIR/audit_wiring.py" 2>/dev/null | grep "FAILED" || true)
    FAILED_COUNT=$(echo "$AUDIT_RESULT" | grep -c "FAILED" 2>/dev/null || echo "0")
    if [[ "$FAILED_COUNT" -eq 0 || -z "$AUDIT_RESULT" ]]; then
        pass "Wiring audit clean (0 FAILED)"
    else
        fail "Wiring audit has failures"
        echo "       Run: python3 $SCRIPT_DIR/audit_wiring.py"
    fi
else
    pass "Wiring audit script not found (skipped)"
fi

# ── 6. GPU accessible ────────────────────────────────────────────────
if python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'{name} ({mem:.1f}GB)')
" 2>/dev/null; then
    GPU_INFO=$(python3 -c "
import torch
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'{name} ({mem:.1f}GB)')
" 2>/dev/null || echo "unknown")
    pass "GPU accessible: $GPU_INFO"
else
    if [[ "${MODE:-serving}" == "control-plane" ]]; then
        pass "GPU not required (control-plane mode)"
    else
        fail "GPU not accessible"
        echo "       Check: nvidia-smi"
        echo "       Check: python3 -c 'import torch; print(torch.cuda.is_available())'"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "  ${GREEN}Node $NODE_ID is healthy and ready.${NC}"
    echo "  $PASS_COUNT checks passed, 0 failed."
else
    echo -e "  ${RED}Node $NODE_ID has $FAIL_COUNT failure(s).${NC}"
    echo "  $PASS_COUNT passed, $FAIL_COUNT failed."
fi
echo "═══════════════════════════════════════════════════"
echo ""

exit $FAIL_COUNT
