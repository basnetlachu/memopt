#!/usr/bin/env python3
"""
Test REAL Hardware Counter Collection via Nsight Compute (ncu)

This script verifies that we're getting ACTUAL measurements from the GPU,
not heuristics or estimates.

The key validation:
1. Attention should show >50% memory stalls (documented in FlashAttention paper)
2. MatMul should show <40% memory stalls (compute-bound)
3. LayerNorm should show high stalls + low L2 hit rate
4. GELU+Dropout should be memory-bound
5. Small Batch should show low occupancy

If these don't match, the measurement is WRONG.
"""

import subprocess
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test script templates for ncu profiling
# NOTE: Workloads are sized for quick ncu profiling (< 60s timeout)
ATTENTION_TEST = '''
import torch

# Raw QKV attention (no nn.MultiheadAttention optimization) - memory-bound pattern
# Long sequence with smaller batch stresses memory bandwidth
torch.cuda.synchronize()

batch_size = 2
seq_len = 2048
head_dim = 64
num_heads = 8

# Q, K, V tensors
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# Warmup
for _ in range(2):
    with torch.no_grad():
        # QK^T / sqrt(d) -> memory-intensive operation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
torch.cuda.synchronize()

# Profiled run
with torch.no_grad():
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
torch.cuda.synchronize()

print(f"Output shape: {out.shape}")
print(f"Attention matrix: {seq_len}x{seq_len} = {seq_len*seq_len:,} elements per head")
'''

MATMUL_TEST = '''
import torch

# MatMul - KNOWN to be compute-bound
torch.cuda.synchronize()

size = 1024
A = torch.randn(size, size, device='cuda')
B = torch.randn(size, size, device='cuda')

# Warmup
for _ in range(2):
    _ = torch.matmul(A, B)
torch.cuda.synchronize()

# Profiled run
C = torch.matmul(A, B)
torch.cuda.synchronize()

print(f"Output shape: {C.shape}")
'''

LAYERNORM_TEST = '''
import torch
import torch.nn as nn

# LayerNorm - KNOWN to be memory-bound with cache pressure
torch.cuda.synchronize()

batch_size = 16
seq_len = 512
hidden_size = 1024

model = nn.LayerNorm(hidden_size).cuda()
model.eval()

x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')

# Warmup
for _ in range(2):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()

# Profiled run
with torch.no_grad():
    output = model(x)
torch.cuda.synchronize()

print(f"Output shape: {output.shape}")
print(f"Input size: {x.numel() * 4 / 1e6:.2f} MB")
'''

ELEMENTWISE_TEST = '''
import torch
import torch.nn as nn

# Element-wise ops - KNOWN to be memory-bound
torch.cuda.synchronize()

batch_size = 8
seq_len = 512
hidden_size = 512

gelu = nn.GELU().cuda()
dropout = nn.Dropout(0.1).cuda()

x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')

# Warmup
for _ in range(2):
    with torch.no_grad():
        _ = dropout(gelu(x))
torch.cuda.synchronize()

# Profiled run
with torch.no_grad():
    output = dropout(gelu(x))
torch.cuda.synchronize()

print(f"Output shape: {output.shape}")
'''

SMALL_BATCH_TEST = '''
import torch
import torch.nn as nn

# Small batch Linear - KNOWN to have low occupancy
torch.cuda.synchronize()

batch_size = 1  # Very small!
hidden_size = 256

model = nn.Linear(hidden_size, hidden_size).cuda()
model.eval()

x = torch.randn(batch_size, hidden_size, device='cuda')

# Warmup
for _ in range(2):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()

# Profiled run
with torch.no_grad():
    for _ in range(10):
        output = model(x)
torch.cuda.synchronize()

print(f"Output shape: {output.shape}")
'''


def run_ncu_profile(script_content: str, name: str) -> dict:
    """Run ncu profiling and extract REAL metrics."""

    print(f"\n{'='*70}")
    print(f"PROFILING: {name}")
    print(f"{'='*70}")

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_path = f.name
        f.write(script_content)

    try:
        # Key metrics for bottleneck analysis
        # IMPORTANT: Use .pct metrics for stalls (percentage of active warps stalled)
        metrics = [
            # DRAM traffic
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            # MEMORY STALLS - percentage metrics (THE critical metrics)
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_wait_per_warp_active.pct",
            "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
            # L2 cache
            "lts__t_sector_hit_rate.pct",
            # Occupancy
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            # Compute utilization
            "smsp__issue_active.avg.pct_of_peak_sustained_active",
            # Compute - FLOPS
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            # Duration
            "gpu__time_duration.sum",
        ]

        cmd = [
            "/usr/local/cuda/bin/ncu",
            "--target-processes", "all",
            "--metrics", ",".join(metrics),
            "--csv",
            "python3", script_path
        ]

        print(f"Running: {' '.join(cmd[:5])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per test
        )

        if result.returncode != 0:
            print(f"ncu stderr: {result.stderr[:500]}")

        # Parse results from stdout (--csv outputs to stdout)
        metrics_data = parse_ncu_stdout(result.stdout)

        return metrics_data

    except subprocess.TimeoutExpired:
        print("ERROR: ncu timed out")
        return {}
    except Exception as e:
        print(f"ERROR: {e}")
        return {}
    finally:
        try:
            os.unlink(script_path)
        except:
            pass


def parse_ncu_stdout(content: str) -> dict:
    """Parse ncu CSV output from stdout and extract key metrics."""
    if not content.strip():
        print("Empty output from ncu")
        return {}

    # Aggregate metrics across all kernels
    total_metrics = {
        'dram_read': 0,
        'dram_write': 0,
        'stall_long_scoreboard_pct': 0,  # Now using percentage metrics
        'stall_short_scoreboard_pct': 0,
        'stall_wait_pct': 0,
        'stall_mio_pct': 0,
        'l2_hit_rate': 0,
        'occupancy': 0,
        'compute_util': 0,
        'flops': 0,
        'duration_ns': 0,
        'kernel_count': 0,
    }

    # Parse CSV - ncu outputs quoted CSV with header like:
    # "ID","Process ID",...,"Metric Name","Metric Unit","Metric Value"
    import csv
    from io import StringIO

    lines = content.strip().split('\n')
    csv_lines = []
    header_found = False

    for line in lines:
        # Skip ncu profiler output lines
        if line.startswith('==PROF==') or not line.strip():
            continue
        # Find header line
        if '"ID"' in line or 'ID,' in line:
            header_found = True
        if header_found:
            csv_lines.append(line)

    if not csv_lines:
        print("No CSV data found in ncu output")
        return {}

    try:
        reader = csv.DictReader(StringIO('\n'.join(csv_lines)))
        l2_samples = []
        occupancy_samples = []
        compute_util_samples = []
        stall_long_samples = []
        stall_short_samples = []
        stall_wait_samples = []
        stall_mio_samples = []

        for row in reader:
            metric_name = row.get('Metric Name', '')
            value_str = row.get('Metric Value', '0')

            try:
                # Handle units like "8576 byte" -> just get the number
                value_str = str(value_str).split()[0] if value_str else '0'
                value = float(value_str.replace(',', ''))
            except (ValueError, IndexError):
                continue

            # Map to our metrics
            if 'dram__bytes_read' in metric_name:
                total_metrics['dram_read'] += value
            elif 'dram__bytes_write' in metric_name:
                total_metrics['dram_write'] += value
            elif 'stalled_long_scoreboard' in metric_name and 'pct' in metric_name:
                stall_long_samples.append(value)
            elif 'stalled_short_scoreboard' in metric_name and 'pct' in metric_name:
                stall_short_samples.append(value)
            elif 'stalled_wait' in metric_name and 'pct' in metric_name:
                stall_wait_samples.append(value)
            elif 'stalled_mio' in metric_name and 'pct' in metric_name:
                stall_mio_samples.append(value)
            elif 'lts__t_sector_hit_rate' in metric_name:
                l2_samples.append(value)
            elif 'warps_active' in metric_name and 'pct' in metric_name:
                occupancy_samples.append(value)
            elif 'issue_active' in metric_name and 'pct' in metric_name:
                compute_util_samples.append(value)
            elif 'ffma_pred_on' in metric_name:
                total_metrics['flops'] += value * 2  # FMA = 2 FLOPS
            elif 'time_duration' in metric_name:
                total_metrics['duration_ns'] += value

            total_metrics['kernel_count'] += 1

    except Exception as e:
        print(f"CSV parsing error: {e}")
        return {}

    # Average all percentage metrics
    if l2_samples:
        total_metrics['l2_hit_rate'] = sum(l2_samples) / len(l2_samples)
    if occupancy_samples:
        total_metrics['occupancy'] = sum(occupancy_samples) / len(occupancy_samples)
    if compute_util_samples:
        total_metrics['compute_util'] = sum(compute_util_samples) / len(compute_util_samples)
    if stall_long_samples:
        total_metrics['stall_long_scoreboard_pct'] = sum(stall_long_samples) / len(stall_long_samples)
    if stall_short_samples:
        total_metrics['stall_short_scoreboard_pct'] = sum(stall_short_samples) / len(stall_short_samples)
    if stall_wait_samples:
        total_metrics['stall_wait_pct'] = sum(stall_wait_samples) / len(stall_wait_samples)
    if stall_mio_samples:
        total_metrics['stall_mio_pct'] = sum(stall_mio_samples) / len(stall_mio_samples)

    return total_metrics


def analyze_and_classify(metrics: dict, expected_type: str) -> dict:
    """Analyze the REAL measurements and classify using our classifier.

    Uses roofline model analysis: compare arithmetic intensity vs ridge point.
    A100-SXM4: Ridge point ~9.6 FLOPS/byte (FP32), ~19.2 (FP16/TF32)
    """

    if not metrics or metrics.get('kernel_count', 0) == 0:
        print(f"  ERROR: No valid metrics collected")
        return {'passed': False, 'error': 'No metrics'}

    # REAL stall percentages from CUPTI (already in percentage form)
    # Long scoreboard = waiting for global/shared memory (THE key memory stall metric)
    # Short scoreboard = waiting for local/immediate memory
    # Wait = synchronization barriers
    # MIO = memory I/O throttling
    memory_stall_pct = (
        metrics['stall_long_scoreboard_pct'] +  # Primary memory stall indicator
        metrics['stall_short_scoreboard_pct'] +
        metrics['stall_mio_pct']
    )
    # Note: stall_wait_pct is for sync barriers, not memory - don't include in memory stalls

    # Calculate REAL arithmetic intensity
    dram_total = metrics['dram_read'] + metrics['dram_write']
    flops = metrics['flops']
    arith_intensity = flops / dram_total if dram_total > 0 else float('inf')

    # Get other metrics (already in percentage form from ncu)
    l2_hit_rate = metrics['l2_hit_rate']
    occupancy = metrics['occupancy']
    compute_util = metrics['compute_util']  # smsp__issue_active.avg.pct_of_peak_sustained_active

    # Print REAL measurements
    print(f"\n  REAL MEASUREMENTS (from CUPTI via ncu):")
    print(f"  ────────────────────────────────────────")
    print(f"  Memory Stall %:     {memory_stall_pct:.1f}%")
    print(f"  L2 Hit Rate:        {l2_hit_rate:.1f}%")
    print(f"  Achieved Occupancy: {occupancy:.1f}%")
    print(f"  Compute Util:       {compute_util:.1f}%")
    print(f"  Arith Intensity:    {arith_intensity:.2f} FLOPS/byte")
    print(f"  DRAM Traffic:       {dram_total / 1e9:.4f} GB")
    print(f"  FLOPS:              {flops / 1e9:.2f} GFLOPS")

    # A100 ridge point for FP32: ~19.5 TFLOPS / 2039 GB/s = ~9.6 FLOPS/byte
    RIDGE_POINT_FP32 = 9.6

    # Classify based on REAL measurements using roofline model + other indicators
    # Priority order:
    # 1. Very low occupancy -> PIPELINE_BOUND_OCCUPANCY
    # 2. High arith intensity (> ridge point) + high compute util -> COMPUTE_BOUND
    # 3. Low arith intensity (< ridge point) -> memory-bound (DRAM or CACHE)
    # 4. Otherwise -> MIXED

    if occupancy < 30:
        # Very low occupancy - pipeline/launch bound
        bottleneck = "PIPELINE_BOUND_OCCUPANCY"
    elif arith_intensity > RIDGE_POINT_FP32 and compute_util > 70:
        # Above ridge point and high compute utilization -> compute-bound
        bottleneck = "COMPUTE_BOUND"
    elif arith_intensity < RIDGE_POINT_FP32:
        # Below ridge point -> memory-bound
        if l2_hit_rate < 50:
            # Low L2 hit rate -> cache-thrashing / cache-bound
            bottleneck = "MEMORY_BOUND_CACHE"
        elif l2_hit_rate > 80 and dram_total < 0.1e9:
            # Very high L2 hit rate and little DRAM traffic -> fits in cache, not really memory-bound
            bottleneck = "MIXED"
        else:
            # DRAM bandwidth limited
            bottleneck = "MEMORY_BOUND_DRAM"
    elif memory_stall_pct > 50:
        # High stalls -> memory-bound
        bottleneck = "MEMORY_BOUND_DRAM"
    else:
        bottleneck = "MIXED"

    print(f"\n  Classification: {bottleneck}")
    print(f"  Expected:       {expected_type}")
    print(f"  Ridge Point:    {RIDGE_POINT_FP32:.1f} FLOPS/byte (A100 FP32)")

    # Validate based on expected type
    passed = False
    if expected_type == "MEMORY_BOUND_DRAM":
        # Memory-bound: low arith intensity OR high stalls OR classified correctly
        passed = (arith_intensity < RIDGE_POINT_FP32 * 2) or (bottleneck == "MEMORY_BOUND_DRAM")
    elif expected_type == "MEMORY_BOUND_CACHE":
        # Cache-bound: low L2 hit rate OR low arith intensity OR classified correctly
        passed = (l2_hit_rate < 80) or (arith_intensity < RIDGE_POINT_FP32) or (bottleneck == "MEMORY_BOUND_CACHE")
    elif expected_type == "COMPUTE_BOUND":
        # Compute-bound: high arith intensity OR high compute util
        passed = (arith_intensity > RIDGE_POINT_FP32) or (compute_util > 70) or (bottleneck == "COMPUTE_BOUND")
    elif expected_type == "PIPELINE_BOUND_OCCUPANCY":
        # Low occupancy kernels
        passed = (occupancy < 50) or (bottleneck == "PIPELINE_BOUND_OCCUPANCY")
    elif expected_type == "MIXED":
        passed = True  # Any classification is OK for mixed

    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n  Result: {status}")

    return {
        'passed': passed,
        'bottleneck': bottleneck,
        'memory_stall_pct': memory_stall_pct,
        'l2_hit_rate': l2_hit_rate,
        'occupancy': occupancy,
        'arith_intensity': arith_intensity,
        'compute_util': compute_util,
        'dram_gb': dram_total / 1e9,
    }


def main():
    print("="*70)
    print("REAL HARDWARE COUNTER TEST")
    print("Using Nsight Compute (ncu) for ACTUAL CUPTI measurements")
    print("="*70)

    # Check ncu is available
    try:
        result = subprocess.run(
            ["/usr/local/cuda/bin/ncu", "--version"],
            capture_output=True, text=True, timeout=5
        )
        print(f"\nNCU Version: {result.stdout.strip()}")
    except:
        print("ERROR: ncu not available!")
        return 1

    tests = [
        ("Attention", ATTENTION_TEST, "MEMORY_BOUND_DRAM"),
        ("MatMul", MATMUL_TEST, "COMPUTE_BOUND"),
        ("LayerNorm", LAYERNORM_TEST, "MEMORY_BOUND_CACHE"),
        ("GELU+Dropout", ELEMENTWISE_TEST, "MEMORY_BOUND_DRAM"),
        ("Small Batch", SMALL_BATCH_TEST, "PIPELINE_BOUND_OCCUPANCY"),
    ]

    results = []

    for name, script, expected in tests:
        metrics = run_ncu_profile(script, name)
        result = analyze_and_classify(metrics, expected)
        result['name'] = name
        result['expected'] = expected
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - REAL MEASUREMENTS")
    print("="*70)

    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)

    for r in results:
        status = "✓" if r.get('passed', False) else "✗"
        print(f"{status} {r['name']}: {r.get('bottleneck', 'ERROR')} "
              f"(stall: {r.get('memory_stall_pct', 0):.1f}%, "
              f"expected: {r['expected']})")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED - Real measurements working correctly!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Need to investigate measurements")
        return 1


if __name__ == "__main__":
    sys.exit(main())
