#!/usr/bin/env bash
#
# Build libmemopt_io.so and libmemopt_gds.so.
# Run from repo root: bash memopt/vmm/io/build.sh
#
# Exit codes:
#   0  both libraries built (GDS may be stub)
#   1  g++ or nvcc not found
#
set -e

IO_SRC="memopt/vmm/io/memopt_io.cpp"
GDS_SRC="memopt/vmm/io/memopt_gds.cpp"
IO_OUT="memopt/vmm/io/libmemopt_io.so"
GDS_OUT="memopt/vmm/io/libmemopt_gds.so"

if ! command -v g++ &>/dev/null; then
    echo "[io/build] ERROR: g++ not found. Install build-essential." >&2
    exit 1
fi

# ── io_uring reader ───────────────────────────────────────────────
echo "[io/build] Checking for liburing..."
URING_FLAGS=""
if pkg-config --exists liburing 2>/dev/null; then
    URING_FLAGS="-luring"
    echo "[io/build] liburing found — io_uring path enabled"
else
    echo "[io/build] liburing not found — pread fallback only"
fi

g++ -O3 -std=c++17 -shared -fPIC \
    ${URING_FLAGS} \
    -o "${IO_OUT}" "${IO_SRC}"
echo "[io/build] Built: ${IO_OUT}"

# ── GPUDirect Storage ─────────────────────────────────────────────
echo "[io/build] Checking for cuFile (GPUDirect Storage)..."
GDS_COMPILED=0

if [ -f "/usr/local/cuda/include/cufile.h" ] && \
   command -v nvcc &>/dev/null; then
    echo "[io/build] cuFile found — building real GDS path"
    nvcc -O3 -shared -Xcompiler -fPIC \
         -lcufile \
         -o "${GDS_OUT}" "${GDS_SRC}" && GDS_COMPILED=1 || true
fi

if [ "${GDS_COMPILED}" -eq 0 ]; then
    echo "[io/build] Building GDS stub (returns ENOTSUP)"
    g++ -O3 -std=c++17 -shared -fPIC \
        -o "${GDS_OUT}" "${GDS_SRC}"
fi

echo "[io/build] Built: ${GDS_OUT}"
echo "[io/build] Done. Verify with:"
echo "  python3 -c 'from memopt.vmm.io import fast_read_block, status; print(status())'"
