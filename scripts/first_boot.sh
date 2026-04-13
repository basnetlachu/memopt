#!/usr/bin/env bash
# memopt first-boot configuration.
#
# Runs exactly once on a golden-image node's first boot.
# Everything else is pre-baked — this is the only script that runs
# on a fresh boot. Subsequent boots short-circuit via the marker file.
#
# Never fails fatally: every step has a fallback so a transient
# problem (e.g. control plane temporarily down) does not brick the
# node.
#
# Wire into systemd via a oneshot unit that runs before
# memopt-transport.service / memopt-serving.service.

set -euo pipefail

FIRST_BOOT_MARKER="/etc/memopt/.first_boot_done"
CONFIG_DIR="/etc/memopt"
DATA_DIR="/var/memopt"
LOG_FILE="${DATA_DIR}/logs/first_boot.log"

mkdir -p "${CONFIG_DIR}" "${DATA_DIR}/logs" "${DATA_DIR}/certs" "${DATA_DIR}/nvme"

if [[ -f "${FIRST_BOOT_MARKER}" ]]; then
    echo "memopt: first boot already completed"
    exit 0
fi

echo "memopt: first boot configuration..."
: > "${LOG_FILE}"

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" | tee -a "${LOG_FILE}"
}

# ── Step 1: detect node identity ──────────────────────────────────────
NODE_ID="$(python3 -c "
import json
try:
    with open('/run/cloud-init/instance-data.json') as f:
        data = json.load(f)
    node_id = data.get('v1', {}).get('instance_id', '')
    if node_id:
        print(node_id)
except Exception:
    pass
" 2>/dev/null || true)"

if [[ -z "${NODE_ID}" ]]; then
    NODE_ID="${MEMOPT_NODE_ID:-$(hostname)}"
fi

log "node_id=${NODE_ID}"

# ── Step 2: detect rack/pod/region ────────────────────────────────────
# Prefer explicit env, fall back to "unknown". Real DHCP options 224/225
# would be read here with a DHCP client plugin on real hardware.
RACK="${MEMOPT_RACK:-unknown}"
POD="${MEMOPT_POD:-unknown}"
REGION="${MEMOPT_REGION:-unknown}"

log "rack=${RACK} pod=${POD} region=${REGION}"

# ── Step 3: write /etc/memopt/config.env ──────────────────────────────
cat > "${CONFIG_DIR}/config.env" <<ENVEOF
MEMOPT_NODE_ID=${NODE_ID}
MEMOPT_RACK=${RACK}
MEMOPT_POD=${POD}
MEMOPT_REGION=${REGION}
MEMOPT_TRANSPORT=tcp
MEMOPT_CERTIFY_ON_STARTUP=true
MEMOPT_NVME_DIR=${DATA_DIR}/nvme
REDIS_URL=${REDIS_URL:-}
MEMOPT_CONTROL_PLANE_URL=${MEMOPT_CONTROL_PLANE_URL:-${CONTROL_PLANE:-}}
ENVEOF

log "wrote ${CONFIG_DIR}/config.env"

# ── Step 4: detect GPU hardware ───────────────────────────────────────
GPU_COUNT=0
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    GPU_COUNT="${GPU_COUNT:-0}"
fi

cat > "${CONFIG_DIR}/hardware.env" <<ENVEOF
MEMOPT_GPU_COUNT=${GPU_COUNT}
ENVEOF

log "gpus=${GPU_COUNT}"

# ── Step 5: silicon certification ─────────────────────────────────────
CERT_STATUS="SKIPPED"
if [[ "${GPU_COUNT}" -gt 0 ]]; then
    if python3 -m memopt certify \
            --node-id "${NODE_ID}" \
            --output-dir "${DATA_DIR}/certs" \
            >> "${LOG_FILE}" 2>&1; then
        CERT_STATUS="PASSED"
    else
        CERT_STATUS="FAILED"
        log "WARN: silicon certification failed — node will start uncertified"
    fi
fi
echo "${CERT_STATUS}" > "${CONFIG_DIR}/cert_status"
log "cert_status=${CERT_STATUS}"

# ── Step 6: register with control plane ───────────────────────────────
CP_URL="${MEMOPT_CONTROL_PLANE_URL:-${CONTROL_PLANE:-}}"
if [[ -n "${CP_URL}" ]]; then
    curl -sf -X POST \
        "${CP_URL}/api/v1/nodes" \
        -H "Content-Type: application/json" \
        -d "{
            \"node_id\":     \"${NODE_ID}\",
            \"rack\":        \"${RACK}\",
            \"pod\":         \"${POD}\",
            \"gpu_count\":   ${GPU_COUNT},
            \"cert_status\": \"${CERT_STATUS}\"
        }" \
        >> "${LOG_FILE}" 2>&1 \
        || log "WARN: control plane registration failed (non-fatal)"
fi

# ── Step 6b: boot callback (records the boot event) ──────────────────
# Distinct from /api/v1/nodes (inventory) — /boot/callback records
# each successful boot with version + cert + timing. Non-fatal.
if [[ -n "${CP_URL}" ]]; then
    BOOT_TIME_S="$(awk '{print $1}' /proc/uptime 2>/dev/null || echo 0)"
    IMAGE_VERSION="$(cat ${CONFIG_DIR}/version 2>/dev/null || echo unknown)"
    MAC_ADDR="$(cat /sys/class/net/*/address 2>/dev/null | head -1 || echo unknown)"
    curl -sf -X POST \
        "${CP_URL}/boot/callback" \
        -H "Content-Type: application/json" \
        -d "{
            \"node_id\":           \"${NODE_ID}\",
            \"mac_address\":       \"${MAC_ADDR}\",
            \"image_version\":     \"${IMAGE_VERSION}\",
            \"gpu_count\":         ${GPU_COUNT},
            \"cert_status\":       \"${CERT_STATUS}\",
            \"boot_time_seconds\": ${BOOT_TIME_S}
        }" \
        >> "${LOG_FILE}" 2>&1 \
        || log "WARN: boot callback failed (non-fatal)"
fi

# ── Step 7: mark complete ─────────────────────────────────────────────
touch "${FIRST_BOOT_MARKER}"
echo "memopt: first boot complete"
echo "  node_id: ${NODE_ID}"
echo "  rack:    ${RACK}"
echo "  pod:     ${POD}"
echo "  gpus:    ${GPU_COUNT}"
echo "  cert:    ${CERT_STATUS}"

# ── Step 8: start services ────────────────────────────────────────────
# Enable units on first boot (docker build couldn't) and start them.
# All are non-fatal so the marker is set even if a service is masked
# or the host is using a different init system in dev.
systemctl enable memopt-transport.service 2>/dev/null || true
systemctl enable memopt-serving.service   2>/dev/null || true
systemctl start  memopt-transport.service 2>/dev/null || true
systemctl start  memopt-serving.service   2>/dev/null || true
