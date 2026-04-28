#!/bin/bash
# Pillar 1 complete test on GPU server.
# Run: bash scripts/deploy_and_test_pillar1.sh
# Requires: CUDA, cmake, Python3, PyTorch

set -e
REPO=/opt/memopt
cd $REPO

echo "========================================"
echo "MEMOPT PILLAR 1 — COMPLETE TEST"
echo "========================================"

echo ""
echo "--- Hardware ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; \
  print('PyTorch:', torch.__version__); \
  print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "--- Installing Python deps ---"
pip install pynvml transformers accelerate \
  bitsandbytes --break-system-packages -q

echo ""
echo "--- Building libmemopt_vmm.so ---"
mkdir -p csrc/cuda_vmm/build
cd csrc/cuda_vmm/build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  2>&1 | grep -E "error|warning|CMake"
make -j$(nproc) 2>&1 | grep -E "error|warning|Built"
echo "Build: done"
cd $REPO

echo ""
echo "--- Verifying symbols ---"
nm -D csrc/cuda_vmm/build/libmemopt_vmm.so \
  | grep -E "torch_malloc|torch_free" \
  | awk '{print $3}'
# Must show 2 lines

echo ""
echo "--- C++ allocator test ---"
csrc/cuda_vmm/build/test_vmm_allocator
# Must show: ALL TESTS PASSED

echo ""
echo "--- Pillar 1 proof ---"
PYTHONPATH=$REPO \
MEMOPT_VMM_LIB=$REPO/csrc/cuda_vmm/build/libmemopt_vmm.so \
python3 tests/pillar1_proof.py \
  2>&1 | tee /tmp/pillar1_result.txt

echo ""
echo "--- Final result ---"
python3 -c "
import json
with open('/tmp/pillar1_result.json') as f:
    d = json.load(f)
print(f'WITHOUT memopt: {d[\"wo_max\"]:,} tokens')
print(f'WITH memopt:    {d[\"wm_max\"]:,} tokens')
print(f'Bytes evicted:  {d[\"evicted_gb\"]:.3f} GB')
print(f'PASSED:         {d[\"passed\"]}')
if not d['passed']:
    if d['evicted_gb'] == 0:
        print('ISSUE: evicted_gb=0')
        print('Allocator installed but')
        print('no eviction triggered.')
        print('Try smaller model or')
        print('lower evict threshold.')
    if d['wm_max'] < d['wo_max']:
        print('ISSUE: regression')
        print('Allocator hurting performance.')
"

echo ""
echo "========================================"
echo "PILLAR 1 TEST COMPLETE"
echo "========================================"
