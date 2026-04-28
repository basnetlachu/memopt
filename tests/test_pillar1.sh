#!/bin/bash
# Run this on the GPU server after rsync
set -e
cd /opt/memopt

echo "=== Building ==="
bash csrc/cuda_vmm/build.sh

echo "=== Verifying symbols ==="
nm -D csrc/cuda_vmm/build/libmemopt_vmm.so | \
  grep -E "torch_malloc|torch_free"

echo "=== Running allocator test ==="
csrc/cuda_vmm/build/test_vmm_allocator

echo "=== Running Pillar 1 proof ==="
PYTHONPATH=/opt/memopt \
MEMOPT_VMM_LIB=/opt/memopt/csrc/cuda_vmm/build/libmemopt_vmm.so \
python3 tests/pillar1_proof.py \
  2>&1 | tee /tmp/pillar1_result.txt

echo "=== Result ==="
tail -20 /tmp/pillar1_result.txt
