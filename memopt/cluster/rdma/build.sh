#!/usr/bin/env bash
set -e
SRC="memopt/cluster/rdma/rdma_write.cpp"
OUT="memopt/cluster/rdma/librdma_write.so"

echo "[rdma/build] Checking for libibverbs..."
IB_FLAGS=""
if pkg-config --exists libibverbs 2>/dev/null || \
   [ -f "/usr/include/infiniband/verbs.h" ]; then
    IB_FLAGS="-libverbs"
    echo "[rdma/build] libibverbs found — RDMA path enabled"
else
    echo "[rdma/build] libibverbs not found — send() fallback"
fi

g++ -O3 -std=c++17 -shared -fPIC ${IB_FLAGS} -o "${OUT}" "${SRC}"
echo "[rdma/build] Built: ${OUT}"
