#!/usr/bin/env bash
set -e
SRC="memopt/cluster/fast_lookup/gkd_map.cpp"
OUT="memopt/cluster/fast_lookup/libgkd_map.so"

if ! command -v g++ &>/dev/null; then
    echo "[gkd/build] ERROR: g++ not found." >&2; exit 1
fi

echo "[gkd/build] Checking for abseil..."
ABSL_FLAGS=""
if pkg-config --exists absl 2>/dev/null; then
    ABSL_FLAGS="-labsl_container -labsl_hash"
    echo "[gkd/build] abseil found — flat_hash_map enabled"
else
    echo "[gkd/build] abseil not found — std::unordered_map"
fi

g++ -O3 -std=c++17 -shared -fPIC ${ABSL_FLAGS} -o "${OUT}" "${SRC}"
echo "[gkd/build] Built: ${OUT}"
