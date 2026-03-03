#!/usr/bin/env bash
set -euo pipefail

log()   { echo "[memopt] $(date '+%Y-%m-%d %H:%M:%S') $*"; }
error() { echo "[memopt] ERROR: $*" >&2; exit 1; }

# Hard fail without license key
if [[ -z "${MEMOPT_LICENSE_KEY:-}" ]]; then
    error "No license key provided."
    error "Set MEMOPT_LICENSE_KEY in your .env file."
    error "Get a license at memopt.com"
    exit 1
fi

log "Validating license..."

KEYGEN_ACCOUNT_ID="a44cf991-48c9-435b-bad6-9ed7dd32cc3e"

RESPONSE=$(curl -sf \
    -X POST \
    "https://api.keygen.sh/v1/accounts/${KEYGEN_ACCOUNT_ID}/licenses/actions/validate-key" \
    -H "Content-Type: application/json" \
    -d "{\"meta\": {\"key\": \"${MEMOPT_LICENSE_KEY}\"}}" \
    2>/dev/null || echo '{"meta":{"valid":false,"detail":"network error"}}')

VALID=$(echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('true' if d.get('meta',{}).get('valid') else 'false')
" 2>/dev/null || echo "false")

if [[ "$VALID" != "true" ]]; then
    # Keygen unreachable or key invalid — delegate to Python validator
    # which has a 24-hour grace period for network outages
    GRACE_RESULT=$(python3 -c "
from memopt.license.validator import validate_license, get_license_key
import os
key = os.getenv('MEMOPT_LICENSE_KEY','')
s = validate_license(key)
print('ok' if s.valid else f'FAIL:{s.error}')
" 2>/dev/null || echo "FAIL:validator_error")

    if [[ "$GRACE_RESULT" != "ok" ]]; then
        REASON="${GRACE_RESULT#FAIL:}"
        error "License check failed: $REASON"
        error "Contact support@memopt.com"
        exit 1
    fi
    log "WARNING: Keygen unreachable — running on cached license (grace period active)"
fi

log "License valid."

# API key — get_or_create_key() returns a plain string
python3 -c "
import os
from memopt.auth.api_key import get_or_create_key, mask_key
from pathlib import Path
key_file = Path.home() / '.memopt' / 'api_key'
already_existed = key_file.exists()
key = get_or_create_key()
if not already_existed:
    print('=== SAVE THIS API KEY ===')
    print(f'MEMOPT_API_KEY={key}')
    print('Set on all nodes that connect to this control plane.')
    print('========================')
else:
    print(f'[memopt] API key loaded: {mask_key(key)}')
"

# Start services
log "Starting memopt | Node: ${NODE_NAME:-memopt-node}"

python3 -m uvicorn memopt.control_plane.server:app \
    --host 0.0.0.0 --port "${MEMOPT_PORT:-8080}" \
    --workers 1 --log-level warning &
CP_PID=$!

python3 -m uvicorn memopt.api.server:app \
    --host 0.0.0.0 --port 8000 \
    --workers 1 --log-level warning &
API_PID=$!

# Poll for control plane readiness instead of blind sleep
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${MEMOPT_PORT:-8080}/health" >/dev/null 2>&1; then
        log "Control plane ready (${i}s)"
        break
    fi
    sleep 1
done

python3 -c "
import os
from memopt.daemon.zero_touch import ZeroTouchDaemon, DaemonConfig
config = DaemonConfig(
    scan_interval_seconds=int(os.getenv('MEMOPT_SCAN_INTERVAL','60')),
    auto_apply=os.getenv('MEMOPT_AUTO_APPLY','false').lower()=='true',
    gpu_cost_per_hour=float(os.getenv('MEMOPT_GPU_COST_PER_HOUR','2.50')),
    node_name=os.getenv('NODE_NAME','memopt-node'),
)
ZeroTouchDaemon(config).run()
" &
DAEMON_PID=$!

log "Dashboard: http://localhost:${MEMOPT_PORT:-8080}"
log "Metrics:   http://localhost:8000/metrics"
wait -n $CP_PID $API_PID $DAEMON_PID
log "A process exited — shutting down."
kill $CP_PID $API_PID $DAEMON_PID 2>/dev/null || true
