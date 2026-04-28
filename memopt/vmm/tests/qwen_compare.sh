#!/usr/bin/env bash
# Run Qwen 32B baseline vs memopt, compare max context.
set -u
cd /opt/memopt
export LD_LIBRARY_PATH=/opt/memopt/csrc/cuda_vmm/build:${LD_LIBRARY_PATH:-}

OUT_DIR=${OUT_DIR:-/tmp/qwen_bench}
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo " ROUND 1: BASELINE (default PyTorch allocator)"
echo "=============================================================="
python3 -u memopt/vmm/tests/qwen_run.py baseline "$OUT_DIR/baseline.json" 2>&1 \
  | grep -Ev '_conversion_method|NumPy'
echo ""
echo "=============================================================="
echo " ROUND 2: WITH memopt VMM"
echo "=============================================================="
python3 -u memopt/vmm/tests/qwen_run.py memopt "$OUT_DIR/memopt.json" 2>&1 \
  | grep -Ev '_conversion_method|NumPy'

echo ""
echo "=============================================================="
echo " COMPARISON"
echo "=============================================================="
python3 - <<EOF
import json
b = json.load(open("$OUT_DIR/baseline.json"))
m = json.load(open("$OUT_DIR/memopt.json"))
print(f"baseline max_ctx: {b['max_ctx']:>7d}")
print(f"memopt   max_ctx: {m['max_ctx']:>7d}")
if b['max_ctx'] > 0:
    print(f"uplift:           {m['max_ctx']/b['max_ctx']:.2f}x")
ev = max((r.get('evictions', 0) for r in m['results']), default=0)
dr = max((r.get('pages_dram', 0) for r in m['results']), default=0)
print(f"memopt evictions fired: {ev}")
print(f"memopt max pages in DRAM: {dr}")
EOF
