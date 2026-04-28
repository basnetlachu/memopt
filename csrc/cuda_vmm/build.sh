#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/csrc/cuda_vmm/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
  2>&1 | tee cmake_output.txt
make -j$(nproc) 2>&1 | tee make_output.txt
echo "Build complete."
echo "Library: $BUILD_DIR/libmemopt_vmm.so"
nm -D libmemopt_vmm.so | grep -E \
  "torch_malloc|torch_free|torch_step|torch_stats" \
  || echo "WARNING: torch symbols not found"
