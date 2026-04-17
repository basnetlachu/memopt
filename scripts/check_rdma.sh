#!/bin/bash
#
# Check RDMA/InfiniBand readiness for memopt GUM.
#
# Run on GPU rental nodes before deployment.
# Reports exactly what is available and what needs to be configured.
#
# Usage: ./scripts/check_rdma.sh

set -euo pipefail

echo "memopt RDMA Readiness Check"
echo "==========================="
echo ""

PASS=0
FAIL=0
WARN=0

check() {
    local name="$1"
    local cmd="$2"
    local required="$3"

    if eval "$cmd" &>/dev/null; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        if [ "$required" = "true" ]; then
            echo "[FAIL] $name"
            FAIL=$((FAIL + 1))
        else
            echo "[WARN] $name (optional)"
            WARN=$((WARN + 1))
        fi
    fi
}

# InfiniBand
echo "--- InfiniBand ---"
check "ibverbs library" \
    "ldconfig -p 2>/dev/null | grep -q libibverbs" \
    "false"

check "IB devices present" \
    "ls /dev/infiniband/ 2>/dev/null | grep -q ." \
    "false"

check "ibv_devinfo available" \
    "which ibv_devinfo 2>/dev/null" \
    "false"

if which ibv_devinfo &>/dev/null; then
    echo ""
    echo "IB device status:"
    ibv_devinfo 2>/dev/null | \
        grep -E "hca_id|port_state" | \
        head -20 || true
fi
echo ""

# UCX
echo "--- UCX ---"
check "ucx-py installed" \
    "python3 -c 'import ucp' 2>/dev/null" \
    "false"

check "ucx_info available" \
    "which ucx_info 2>/dev/null" \
    "false"
echo ""

# CUDA GPUDirect RDMA
echo "--- GPUDirect RDMA ---"
check "nvidia-peermem module" \
    "lsmod 2>/dev/null | grep -q nvidia_peermem" \
    "false"
echo ""

# memopt transport
echo "--- memopt Transport ---"
check "memopt-transport binary" \
    "which memopt-transport 2>/dev/null" \
    "false"

check "TCP fallback (always works)" \
    "python3 -c 'import socket; socket.socket()'" \
    "true"
echo ""

# Network check between nodes
echo "--- Network ---"
if [ -n "${MEMOPT_NODE_HOSTS:-}" ]; then
    echo "Testing connectivity to peers..."
    IFS=',' read -ra PEERS <<< "$MEMOPT_NODE_HOSTS"
    for peer in "${PEERS[@]}"; do
        host="${peer%%:*}"
        port="${peer##*:}"
        port="${port:-18516}"
        if timeout 2 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            echo "[PASS] $host:$port reachable"
            PASS=$((PASS + 1))
        else
            echo "[FAIL] $host:$port unreachable"
            FAIL=$((FAIL + 1))
        fi
    done
else
    echo "[WARN] MEMOPT_NODE_HOSTS not set"
    echo "       Set it to test peer connectivity"
    WARN=$((WARN + 1))
fi
echo ""

# Summary
echo "==========================="
echo "PASS: $PASS"
echo "WARN: $WARN"
echo "FAIL: $FAIL"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "STATUS: RDMA not available."
    echo "        TCP fallback will be used."
    echo "        GUM works over TCP."
else
    if [ $WARN -gt 0 ]; then
        echo "STATUS: Partial RDMA support."
    else
        echo "STATUS: RDMA ready."
    fi
fi

echo ""
echo "HONEST NOTE:"
echo "RDMA requires InfiniBand hardware."
echo "Without IB: TCP is used automatically."
echo "GUM works over TCP."
echo "See /gum/stats after deployment for real numbers."
