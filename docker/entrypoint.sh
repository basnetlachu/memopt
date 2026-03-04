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

KEYGEN_ACCOUNT_ID="${KEYGEN_ACCOUNT_ID:-85efe00f-f369-4a5c-95c4-cc1c9a7ebb6a}"

VALID=$(python3 -c "
import sys, os
sys.path.insert(0, '/app')
from memopt.license.validator import validate_license
key = os.environ['MEMOPT_LICENSE_KEY']
s = validate_license(key)
print('true' if s.valid else 'false')
print(s.error or '')
" 2>/dev/null)

VALID_FLAG=$(echo "$VALID" | head -1)
REASON=$(echo "$VALID" | tail -1)

if [[ "$VALID_FLAG" != "true" ]]; then
    error "License invalid: ${REASON:-unknown}"
    error "Contact support@memopt.com"
    exit 1
fi

log "License valid."

# API key — get_or_create_key() returns a plain string
python3 -c "
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
"

# Start services
log "Starting memopt v1.0.0 | Node: ${NODE_NAME:-memopt-node}"

MEMOPT_MODE="${MEMOPT_MODE:-full}"

case "$MEMOPT_MODE" in
  full)
    python3 -m memopt.control_plane.server &
    python3 -m memopt.api.server &
    sleep 3
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
    log "Dashboard: http://localhost:8080"
    log "Metrics:   http://localhost:8000/metrics"
    log "Health:    http://localhost:8080/health"
    wait
    ;;
  daemon)
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
"
    ;;
  control-plane)
    python3 -m memopt.control_plane.server
    ;;
  scan)
    python3 -m memopt.daemon.cli scan
    ;;
  *)
    error "Unknown mode: $MEMOPT_MODE. Valid: full, daemon, control-plane, scan"
    ;;
esac
