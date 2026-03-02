#!/usr/bin/env bash
# memopt container entrypoint
# Handles: license validation, API key setup, TLS setup, service startup
set -euo pipefail

MEMOPT_MODE="${MEMOPT_MODE:-full}"
MEMOPT_PORT="${MEMOPT_PORT:-8080}"
MEMOPT_HTTPS="${MEMOPT_HTTPS:-false}"
MEMOPT_VERSION="${MEMOPT_VERSION:-1.0.0}"

log() { echo "[memopt] $(date '+%Y-%m-%d %H:%M:%S') $*"; }
error() { log "ERROR: $*" >&2; exit 1; }

# ── Step 1: Validate license ──────────────────────────────────────────────────
validate_license() {
    if [[ -z "${MEMOPT_LICENSE_KEY:-}" ]]; then
        error "MEMOPT_LICENSE_KEY not set. Get your key at memopt.ai"
    fi

    log "Validating license key..."

    KEYGEN_ACCOUNT_ID="${KEYGEN_ACCOUNT_ID:-YOUR_KEYGEN_ACCOUNT_ID}"
    RESPONSE=$(curl -sf --max-time 10 \
        -X POST \
        "https://api.keygen.sh/v1/accounts/${KEYGEN_ACCOUNT_ID}/licenses/actions/validate-key" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "{\"meta\": {\"key\": \"${MEMOPT_LICENSE_KEY}\"}}" \
        2>/dev/null || echo '{"meta":{"valid":false},"errors":[{"title":"Network error"}]}')

    VALID_RESULT=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    meta = d.get('meta', {})
    is_valid = 'true' if meta.get('valid') else 'false'
    data = d.get('data', {})
    attrs = data.get('attributes', {}) if data else {}
    metadata = attrs.get('metadata', {}) if attrs else {}
    gpu_limit = metadata.get('gpu_limit', '4')
    tier = metadata.get('tier', 'free')
    print(is_valid)
    print(gpu_limit)
    print(tier)
except Exception as e:
    print('false')
    print('4')
    print('free')
" 2>/dev/null)

    VALID_STATUS=$(echo "$VALID_RESULT" | sed -n '1p')
    GPU_LIMIT=$(echo "$VALID_RESULT" | sed -n '2p')
    TIER=$(echo "$VALID_RESULT" | sed -n '3p')

    if [[ "$VALID_STATUS" != "true" ]]; then
        # Allow running in offline/grace-period mode — Python validator handles this
        log "WARNING: Could not validate license with Keygen — running in grace-period mode"
        log "If this persists beyond 24 hours, memopt will stop. Contact support@memopt.ai"
    else
        log "License valid. Tier: ${TIER} | GPU entitlement: ${GPU_LIMIT}"
    fi

    export MEMOPT_GPU_LIMIT="${GPU_LIMIT:-4}"
    export MEMOPT_TIER="${TIER:-free}"
}

# ── Step 2: Generate or load API key ─────────────────────────────────────────
setup_api_key() {
    API_KEY_DIR="${HOME}/.memopt"
    API_KEY_FILE="${API_KEY_DIR}/api_key"
    mkdir -p "$API_KEY_DIR"

    if [[ -n "${MEMOPT_API_KEY:-}" ]]; then
        log "Using provided MEMOPT_API_KEY"
        echo "$MEMOPT_API_KEY" > "$API_KEY_FILE"
        chmod 600 "$API_KEY_FILE"
    elif [[ -f "$API_KEY_FILE" ]]; then
        log "Using existing API key from $API_KEY_FILE"
    else
        log "Generating new API key..."
        python3 -c "
from memopt.auth.api_key import get_or_create_key, mask_key
key = get_or_create_key()
print('=== SAVE THIS API KEY ===')
print(f'MEMOPT_API_KEY={key}')
print('Set this env var on all nodes that connect to this control plane.')
print('========================')
"
    fi
}

# ── Step 3: Setup TLS if requested ───────────────────────────────────────────
setup_tls() {
    if [[ "$MEMOPT_HTTPS" != "true" ]]; then
        return
    fi

    TLS_DIR="${HOME}/.memopt/tls"
    mkdir -p "$TLS_DIR"
    CERT_FILE="${TLS_DIR}/memopt.crt"
    KEY_FILE="${TLS_DIR}/memopt.key"

    if [[ -f "$CERT_FILE" && -f "$KEY_FILE" ]]; then
        log "Using existing TLS certificates from $TLS_DIR"
    elif [[ -n "${MEMOPT_TLS_CERT:-}" && -n "${MEMOPT_TLS_KEY:-}" ]]; then
        log "Using TLS certificates from environment variables"
        echo "$MEMOPT_TLS_CERT" > "$CERT_FILE"
        echo "$MEMOPT_TLS_KEY" > "$KEY_FILE"
        chmod 600 "$KEY_FILE"
    else
        log "Generating self-signed TLS certificate for ${MEMOPT_DOMAIN:-localhost}..."
        openssl req -x509 -newkey rsa:4096 \
            -keyout "$KEY_FILE" \
            -out "$CERT_FILE" \
            -days 3650 \
            -nodes \
            -subj "/CN=${MEMOPT_DOMAIN:-localhost}/O=memopt/C=US" \
            -addext "subjectAltName=DNS:${MEMOPT_DOMAIN:-localhost}" \
            -quiet 2>/dev/null
        chmod 600 "$KEY_FILE"
        log "Self-signed cert generated. For production, provide your own certs."
    fi

    # Render nginx TLS config from template
    DOMAIN="${MEMOPT_DOMAIN:-localhost}"
    sed \
        -e "s|\${DOMAIN}|${DOMAIN}|g" \
        -e "s|\${CERT_PATH}|${CERT_FILE}|g" \
        -e "s|\${KEY_PATH}|${KEY_FILE}|g" \
        -e "s|\${UPSTREAM}|127.0.0.1:${MEMOPT_PORT}|g" \
        /etc/nginx/memopt.conf.template > /etc/nginx/sites-enabled/memopt-tls

    nginx -t 2>/dev/null && nginx -s reload 2>/dev/null || nginx
    log "TLS enabled at https://${DOMAIN}"
}

# ── Step 4: Start services ────────────────────────────────────────────────────
start_services() {
    case "$MEMOPT_MODE" in
        full)
            log "Starting full memopt stack..."
            log "  Control plane:  port $MEMOPT_PORT"
            log "  Per-node API:   port 8000"
            log "  Zero-touch daemon: enabled"

            # Start control plane (dashboard + fleet management)
            python3 -m uvicorn memopt.control_plane.server:app \
                --host 0.0.0.0 --port "$MEMOPT_PORT" --workers 1 \
                --log-level warning &
            CP_PID=$!

            # Start per-node REST API + Prometheus metrics
            python3 -m uvicorn memopt.api.server:app \
                --host 0.0.0.0 --port 8000 --workers 1 \
                --log-level warning &
            API_PID=$!

            # Wait for control plane to be ready (up to 30s)
            log "Waiting for control plane to be ready..."
            for i in $(seq 1 30); do
                if curl -sf "http://localhost:$MEMOPT_PORT/health" > /dev/null 2>&1; then
                    log "Control plane ready."
                    break
                fi
                sleep 1
            done

            # Start zero-touch daemon
            python3 -c "
import os
from memopt.daemon.zero_touch import ZeroTouchDaemon, DaemonConfig
config = DaemonConfig(
    scan_interval_seconds=int(os.getenv('MEMOPT_SCAN_INTERVAL', '60')),
    sample_seconds=int(os.getenv('MEMOPT_SAMPLE_SECONDS', '5')),
    auto_apply=os.getenv('MEMOPT_AUTO_APPLY', 'false').lower() == 'true',
    gpu_cost_per_hour=float(os.getenv('MEMOPT_GPU_COST_PER_HOUR', '2.50')),
    node_name=os.getenv('NODE_NAME', 'memopt-node'),
)
daemon = ZeroTouchDaemon(config)
daemon.run()
" &
            DAEMON_PID=$!

            log "memopt stack running:"
            log "  Dashboard:    http://localhost:$MEMOPT_PORT"
            log "  Metrics:      http://localhost:8000/metrics"
            log "  Health:       http://localhost:$MEMOPT_PORT/health"

            # Exit if any core process dies
            wait -n $CP_PID $API_PID $DAEMON_PID
            log "A core process exited — shutting down."
            kill $CP_PID $API_PID $DAEMON_PID 2>/dev/null || true
            ;;

        daemon)
            log "Starting daemon-only mode"
            log "Reporting to: ${MEMOPT_CONTROL_PLANE:-<standalone>}"
            python3 -c "
import os
from memopt.daemon.zero_touch import ZeroTouchDaemon, DaemonConfig
config = DaemonConfig(
    scan_interval_seconds=int(os.getenv('MEMOPT_SCAN_INTERVAL', '60')),
    sample_seconds=int(os.getenv('MEMOPT_SAMPLE_SECONDS', '5')),
    auto_apply=os.getenv('MEMOPT_AUTO_APPLY', 'false').lower() == 'true',
    gpu_cost_per_hour=float(os.getenv('MEMOPT_GPU_COST_PER_HOUR', '2.50')),
    node_name=os.getenv('NODE_NAME', 'memopt-node'),
)
daemon = ZeroTouchDaemon(config)
daemon.run()
"
            ;;

        control-plane)
            log "Starting control plane only on port $MEMOPT_PORT"
            python3 -m uvicorn memopt.control_plane.server:app \
                --host 0.0.0.0 --port "$MEMOPT_PORT" --workers 1
            ;;

        scan)
            # One-shot scan — useful for quick testing
            python3 -m memopt.daemon.cli scan
            ;;

        *)
            error "Unknown MEMOPT_MODE: $MEMOPT_MODE. Use: full | daemon | control-plane | scan"
            ;;
    esac
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    log "memopt v${MEMOPT_VERSION} starting"
    log "Mode:   $MEMOPT_MODE"
    log "Node:   ${NODE_NAME:-memopt-node}"

    validate_license
    setup_api_key
    setup_tls
    start_services
}

case "${1:-start}" in
    start)
        main
        ;;
    scan)
        validate_license
        python3 -m memopt.daemon.cli scan
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        exec "$@"
        ;;
esac
