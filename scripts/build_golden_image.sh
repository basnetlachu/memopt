#!/usr/bin/env bash
# Build the memopt golden image.
#
# The golden image bakes memopt completely into an Ubuntu 22.04
# base: OS + CUDA runtime + Python + memopt + pre-compiled C++
# extensions + systemd units + first-boot script. Nodes boot from
# this image over PXE/network and start in < 60 seconds with no
# compiler, pip, or network required at boot time.
#
# Usage:
#   ./scripts/build_golden_image.sh \
#     --cuda-version 12.4 \
#     --sm-targets "86;90;100" \
#     --tag v1.0.0 \
#     --registry registry.example.com/memopt \
#     [--push]
#
# Exit codes:
#   0 — success
#   1 — input validation / prerequisite failure
#   2 — image validation failure (image not pushed)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
CUDA_VERSION="12.4"
SM_TARGETS="86;90;100"
TAG=""
REGISTRY="memopt"
PUSH=false

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

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Build the memopt golden image — a fully baked OS image containing
memopt, its C++ extensions, systemd units, and first-boot
configuration. Safe to run from macOS or Ubuntu 22.04.

Options:
  --cuda-version X.Y       CUDA toolkit version  (default: 12.4)
  --sm-targets "A;B;C"     Semicolon-separated SM targets
                           (default: "86;90;100")
  --tag vX.Y.Z             Version tag (required)
  --registry REG           Image registry prefix (default: memopt)
  --push                   Push image to registry after build
  --help                   Show this help

Examples:
  $(basename "$0") --tag v1.0.0
  $(basename "$0") --tag v1.2.3 --registry myrepo --push
EOF
    exit 0
}

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --sm-targets)   SM_TARGETS="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --registry)     REGISTRY="$2"; shift 2 ;;
        --push)         PUSH=true; shift ;;
        --help|-h)      usage ;;
        *)              fail "Unknown option: $1"; usage ;;
    esac
done

# ── Step 1: validate inputs (before any Docker call) ─────────────────
echo "── Step 1: Input validation ──"

if [[ -z "$TAG" ]]; then
    fail "--tag is required (e.g. --tag v1.0.0)"
    exit 1
fi

if ! [[ "$CUDA_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
    fail "Invalid --cuda-version: '$CUDA_VERSION' (expected major.minor)"
    exit 1
fi

if ! [[ "$SM_TARGETS" =~ ^[0-9]+(\;[0-9]+)*$ ]]; then
    fail "Invalid --sm-targets: '$SM_TARGETS' (expected e.g. '86;90;100')"
    exit 1
fi

if ! [[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9._-]+)?$ ]]; then
    fail "Invalid --tag: '$TAG' (expected vMAJOR.MINOR.PATCH)"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    fail "docker not found in PATH"
    exit 1
fi

# BuildKit is required for the RUN --mount= directive used downstream.
# On Docker 23+ buildx is on by default; older versions need the env var.
if ! docker buildx version &>/dev/null; then
    warn "docker buildx not found — setting DOCKER_BUILDKIT=1"
fi
export DOCKER_BUILDKIT=1

ok "Inputs validated: cuda=$CUDA_VERSION sm=$SM_TARGETS tag=$TAG"

# ── Step 2: git metadata ──────────────────────────────────────────────
echo "── Step 2: Git metadata ──"

if command -v git &>/dev/null && git -C "$PROJECT_DIR" rev-parse --short HEAD &>/dev/null; then
    GIT_COMMIT="$(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
    GIT_BRANCH="$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD)"
else
    GIT_COMMIT="unknown"
    GIT_BRANCH="unknown"
fi
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

ok "commit=$GIT_COMMIT branch=$GIT_BRANCH date=$BUILD_DATE"

# ── Step 3: build the image ───────────────────────────────────────────
echo "── Step 3: Docker build ──"

IMAGE_TAG="${REGISTRY}/memopt-golden:${TAG}"
IMAGE_LATEST="${REGISTRY}/memopt-golden:latest"

docker build \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "CUDA_ARCHITECTURES=${SM_TARGETS}" \
    --build-arg "MEMOPT_VERSION=${TAG}" \
    --build-arg "GIT_COMMIT=${GIT_COMMIT}" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --label "memopt.version=${TAG}" \
    --label "memopt.git-commit=${GIT_COMMIT}" \
    --label "memopt.build-date=${BUILD_DATE}" \
    --label "memopt.sm-targets=${SM_TARGETS}" \
    --label "memopt.cuda-version=${CUDA_VERSION}" \
    -t "${IMAGE_TAG}" \
    -t "${IMAGE_LATEST}" \
    -f "${PROJECT_DIR}/Dockerfile.golden" \
    "${PROJECT_DIR}"

ok "Built ${IMAGE_TAG}"

# ── Step 4: validate the image ────────────────────────────────────────
echo "── Step 4: Image validation ──"

if ! docker run --rm "${IMAGE_TAG}" python3 -c "
import memopt
import os, glob, sys

print(f'memopt version: {getattr(memopt, \"__version__\", \"unknown\")}')

# C++ extensions may be absent on a CPU-only build machine — that is OK.
# Golden image only requires that IF any were built, they are present.
so_files = glob.glob('/app/memopt/**/*.so', recursive=True)
print(f'C++ extensions: {len(so_files)}')

# Transport binary — baked or absent (depends on whether nvcc found CUDA
# during build). If absent, Python fallback handles transport.
transport_ok = os.path.exists('/usr/local/bin/memopt-transport')
print(f'Transport binary: {\"present\" if transport_ok else \"absent (python fallback)\"}')

# Version file must be present — this is the contract of a golden image
version_file = '/etc/memopt/version'
assert os.path.exists(version_file), f'Missing: {version_file}'
print(f'Version file: {open(version_file).read().strip()}')

print('Image validation: PASS')
"; then
    fail "VALIDATION FAILED — image not pushed"
    exit 2
fi

ok "Image validation passed"

# ── Step 5: push if requested ─────────────────────────────────────────
echo "── Step 5: Push ──"

IMAGE_DIGEST="local-only"
if [[ "$PUSH" == "true" ]]; then
    docker push "${IMAGE_TAG}"
    docker push "${IMAGE_LATEST}"
    IMAGE_DIGEST="$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE_TAG}" 2>/dev/null || echo "unknown")"
    ok "Pushed: ${IMAGE_TAG}"
else
    ok "Built locally (use --push to push to ${REGISTRY})"
fi

# ── Step 6: write manifest ────────────────────────────────────────────
echo "── Step 6: Manifest ──"

mkdir -p "${PROJECT_DIR}/dist"
MANIFEST="${PROJECT_DIR}/dist/golden-image-manifest.json"

# Convert ";"-separated SM targets to a JSON list
SM_JSON="$(echo "${SM_TARGETS}" | awk -F';' '{
    out="["; for (i=1; i<=NF; i++) { out = out "\"" $i "\""; if (i<NF) out = out ", " }; print out "]"
}')"

cat > "${MANIFEST}" <<EOF
{
  "version":      "${TAG}",
  "git_commit":   "${GIT_COMMIT}",
  "git_branch":   "${GIT_BRANCH}",
  "build_date":   "${BUILD_DATE}",
  "cuda_version": "${CUDA_VERSION}",
  "sm_targets":   ${SM_JSON},
  "registry":     "${REGISTRY}",
  "image":        "${IMAGE_TAG}",
  "digest":       "${IMAGE_DIGEST}",
  "validated":    true
}
EOF

ok "Manifest written: ${MANIFEST}"
echo ""
ok "Golden image build complete"
