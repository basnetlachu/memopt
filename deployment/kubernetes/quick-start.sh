#!/usr/bin/env bash
# memopt Kubernetes Quick Start
# Usage: ./quick-start.sh --license sk-lic-xxxx [--namespace memopt] [--cost 8.00]
set -euo pipefail

LICENSE_KEY=""
NAMESPACE="memopt"
GPU_COST="8.00"
AUTO_APPLY="false"
CHART_VERSION="latest"

log() { echo "[memopt] $*"; }
error() { log "ERROR: $*" >&2; exit 1; }

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --license)       LICENSE_KEY="$2"; shift 2 ;;
        --namespace)     NAMESPACE="$2";   shift 2 ;;
        --cost)          GPU_COST="$2";    shift 2 ;;
        --auto-apply)    AUTO_APPLY="true"; shift ;;
        --chart-version) CHART_VERSION="$2"; shift 2 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ -z "$LICENSE_KEY" ]]; then
    error "--license is required. Get yours at memopt.ai"
fi

command -v kubectl >/dev/null 2>&1 || error "kubectl not found — install kubectl first"
command -v helm    >/dev/null 2>&1 || error "helm not found — install Helm 3.x first"

log "memopt Kubernetes Quick Start"
log "Namespace:  $NAMESPACE"
log "GPU cost:   \$${GPU_COST}/hr"
log "Auto-apply: $AUTO_APPLY"
echo ""

# ── Step 1: Add Helm repo ────────────────────────────────────────────────────
log "Step 1: Adding memopt Helm repository..."
helm repo add memopt https://charts.memopt.ai \
    --username memopt-client \
    --password "$LICENSE_KEY" 2>/dev/null || true
helm repo update
log "Helm repo ready."

# ── Step 2: Create namespace ─────────────────────────────────────────────────
log "Step 2: Creating namespace $NAMESPACE..."
kubectl create namespace "$NAMESPACE" 2>/dev/null || true

# ── Step 3: Create license secret ───────────────────────────────────────────
log "Step 3: Creating license secret..."
kubectl create secret generic memopt-license \
    --namespace "$NAMESPACE" \
    --from-literal=license-key="$LICENSE_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# ── Step 4: Install / upgrade ─────────────────────────────────────────────────
log "Step 4: Installing memopt..."
HELM_ARGS=(
    upgrade --install memopt memopt/memopt
    --namespace "$NAMESPACE"
    --set "license.key=${LICENSE_KEY}"
    --set "daemon.gpuCostPerHour=${GPU_COST}"
    --set "daemon.autoApply=${AUTO_APPLY}"
    --wait
    --timeout 5m
)

if [[ "$CHART_VERSION" != "latest" ]]; then
    HELM_ARGS+=(--version "$CHART_VERSION")
fi

helm "${HELM_ARGS[@]}"

# ── Step 5: Verify ────────────────────────────────────────────────────────────
log "Step 5: Verifying deployment..."
kubectl rollout status deployment/memopt-control-plane \
    --namespace "$NAMESPACE" --timeout=120s

echo ""
log "memopt is running! Next steps:"
echo ""
echo "  # View dashboard:"
echo "  kubectl port-forward -n $NAMESPACE svc/memopt-control-plane 8080:8080"
echo "  open http://localhost:8080"
echo ""
echo "  # View logs:"
echo "  kubectl logs -n $NAMESPACE -l app=memopt -f"
echo ""
echo "  # Check GPU scans:"
echo "  kubectl exec -n $NAMESPACE deploy/memopt-daemon -- memopt scan"
echo ""
echo "  # Get API key (set on all nodes):"
echo "  kubectl logs -n $NAMESPACE deploy/memopt-control-plane | grep 'MEMOPT_API_KEY'"
