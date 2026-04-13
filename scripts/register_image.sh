#!/usr/bin/env bash
# Register a memopt golden image version with the control plane.
# Called by CI/CD after a successful golden image build.
#
# Usage:
#   ./scripts/register_image.sh \
#     --version v1.2.3 \
#     --control-plane http://cp:8765 \
#     --api-key "$MEMOPT_API_KEY" \
#     --git-commit abc1234 \
#     --image-url registry/memopt-golden:v1.2.3 \
#     --cuda-version 12.4 \
#     --sm-targets "86;90;100" \
#     [--build-date 2026-04-13T00:00:00Z] \
#     [--digest sha256:...] \
#     [--mark-stable]
#
# Exit codes:
#   0 — success
#   1 — input validation failure
#   2 — control-plane registration failed

set -euo pipefail

VERSION=""
CONTROL_PLANE=""
API_KEY=""
GIT_COMMIT=""
IMAGE_URL=""
CUDA_VERSION=""
SM_TARGETS=""
BUILD_DATE=""
DIGEST=""
MARK_STABLE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Register a memopt golden image version with the control plane.

Required:
  --version vX.Y.Z             Image version tag
  --control-plane URL          Control plane URL
  --api-key KEY                Control plane API key
  --git-commit SHA             Short git commit
  --image-url URL              Full registry URL to the image
  --cuda-version X.Y           CUDA version
  --sm-targets "A;B;C"         Semicolon-separated SM targets

Optional:
  --build-date ISO             RFC-3339 build timestamp (default: now)
  --digest SHA                 Image content digest
  --mark-stable                Mark as stable after registration
  --help                       Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)       VERSION="$2"; shift 2 ;;
        --control-plane) CONTROL_PLANE="$2"; shift 2 ;;
        --api-key)       API_KEY="$2"; shift 2 ;;
        --git-commit)    GIT_COMMIT="$2"; shift 2 ;;
        --image-url)     IMAGE_URL="$2"; shift 2 ;;
        --cuda-version)  CUDA_VERSION="$2"; shift 2 ;;
        --sm-targets)    SM_TARGETS="$2"; shift 2 ;;
        --build-date)    BUILD_DATE="$2"; shift 2 ;;
        --digest)        DIGEST="$2"; shift 2 ;;
        --mark-stable)   MARK_STABLE=true; shift ;;
        --help|-h)       usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ── Validate inputs ───────────────────────────────────────────────────
missing=()
[[ -z "$VERSION" ]]       && missing+=("--version")
[[ -z "$CONTROL_PLANE" ]] && missing+=("--control-plane")
[[ -z "$API_KEY" ]]       && missing+=("--api-key")
[[ -z "$GIT_COMMIT" ]]    && missing+=("--git-commit")
[[ -z "$IMAGE_URL" ]]     && missing+=("--image-url")
[[ -z "$CUDA_VERSION" ]]  && missing+=("--cuda-version")
[[ -z "$SM_TARGETS" ]]    && missing+=("--sm-targets")

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing required options: ${missing[*]}" >&2
    exit 1
fi

if ! [[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9._-]+)?$ ]]; then
    echo "ERROR: invalid --version '$VERSION' (expected vMAJOR.MINOR.PATCH)" >&2
    exit 1
fi

if [[ -z "$BUILD_DATE" ]]; then
    BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# ── Register ──────────────────────────────────────────────────────────
PAYLOAD="$(python3 -c "
import json, sys
print(json.dumps({
    'version':      '$VERSION',
    'git_commit':   '$GIT_COMMIT',
    'build_date':   '$BUILD_DATE',
    'cuda_version': '$CUDA_VERSION',
    'sm_targets':   '$SM_TARGETS',
    'image_url':    '$IMAGE_URL',
    'digest':       '$DIGEST',
}))
")"

echo "Registering ${VERSION} with ${CONTROL_PLANE}..."
HTTP_CODE="$(curl -sf -o /tmp/register_image_resp.json \
    -w '%{http_code}' \
    -X POST "${CONTROL_PLANE}/api/v1/images" \
    -H "Content-Type: application/json" \
    -H "X-Memopt-API-Key: ${API_KEY}" \
    -d "${PAYLOAD}" || echo "000")"

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "ERROR: registration failed (HTTP $HTTP_CODE)" >&2
    cat /tmp/register_image_resp.json >&2 2>/dev/null || true
    exit 2
fi

echo "Registered: ${VERSION}"
cat /tmp/register_image_resp.json

# ── Mark stable if requested ──────────────────────────────────────────
if [[ "$MARK_STABLE" == "true" ]]; then
    echo "Marking ${VERSION} as stable..."
    HTTP_CODE="$(curl -sf -o /tmp/stable_resp.json \
        -w '%{http_code}' \
        -X POST "${CONTROL_PLANE}/api/v1/images/${VERSION}/stable" \
        -H "X-Memopt-API-Key: ${API_KEY}" || echo "000")"

    if [[ "$HTTP_CODE" != "200" ]]; then
        echo "ERROR: mark-stable failed (HTTP $HTTP_CODE)" >&2
        exit 2
    fi

    echo "Marked stable: ${VERSION}"
fi

echo ""
echo "Image registration complete:"
echo "  version:     ${VERSION}"
echo "  git_commit:  ${GIT_COMMIT}"
echo "  image_url:   ${IMAGE_URL}"
echo "  mark_stable: ${MARK_STABLE}"
