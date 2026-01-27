#!/usr/bin/env python3
"""
Full Integration Test - 3-Step Memory Optimization Pipeline

Tests on real NVIDIA GPU:
1. Continuous Profiler with bottleneck detection
2. Traffic Attribution with optimization synthesis
3. Adaptive Optimizer with feedback loop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

print("=" * 70)
print("FULL INTEGRATION TEST - MEMORY OPTIMIZATION PIPELINE")
print("=" * 70)
print()

# Check environment
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()


# Create test model
class TransformerBlock(nn.Module):
    def __init__(self, dim=512, heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TestModel(nn.Module):
    def __init__(self, dim=512, depth=6, heads=8):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TestModel(dim=512, depth=6, heads=8).to(device)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model: {param_count/1e6:.1f}M parameters")
print()

# Sample input
batch_size = 8
seq_len = 128
sample_input = torch.randn(batch_size, seq_len, 512).to(device)


# ============================================================
# STEP 1: CONTINUOUS PROFILER
# ============================================================
print("=" * 70)
print("STEP 1: CONTINUOUS PROFILER + BOTTLENECK DETECTION")
print("=" * 70)
print()

from memopt.profiler import ContinuousProfiler

profiler = ContinuousProfiler()
profiler.start()

# Profile multiple iterations
for i in range(10):
    with profiler.profile_region(f"forward_{i}"):
        with torch.no_grad():
            output = model(sample_input)
        if device == 'cuda':
            torch.cuda.synchronize()

profiler.stop()
snapshot = profiler.snapshot()

print(f"Profiled Regions: {len(snapshot.kernel_analyses)}")
print(f"Total GPU Time: {snapshot.total_gpu_time_ms:.2f} ms")
print(f"Total DRAM Traffic: {snapshot.total_dram_bytes / 1024**2:.2f} MB")
print(f"Memory-Bound Time: {snapshot.memory_bound_pct:.1f}%")
print()

if snapshot.top_bottlenecks:
    print("Top Bottlenecks:")
    for i, b in enumerate(snapshot.top_bottlenecks[:5], 1):
        print(f"  {i}. {b.kernel_name}")
        print(f"     Type: {b.bottleneck_type.value}")
        print(f"     Impact Score: {b.impact_score:.4f}")
        print(f"     Memory Stall: {b.memory_stall_pct:.1f}%")
        if b.recommendations:
            print(f"     Recommendation: {b.recommendations[0]}")
print()


# ============================================================
# STEP 2: TRAFFIC ATTRIBUTION
# ============================================================
print("=" * 70)
print("STEP 2: TRAFFIC ATTRIBUTION + OPTIMIZATION SYNTHESIS")
print("=" * 70)
print()

from memopt.profiler import TrafficAttributor

attributor = TrafficAttributor()
attributor.analyze_model(model, sample_input)

attributions = attributor.get_attributions()
candidates = attributor.get_optimization_candidates()

print(f"Attributions Found: {len(attributions)}")
for i, attr in enumerate(attributions[:3], 1):
    print(f"  {i}. {attr.attribution_type.value}")
    print(f"     Confidence: {attr.confidence:.0%}")
    print(f"     Affected Tensors: {len(attr.affected_tensors)}")
    print(f"     Estimated Reduction: {attr.estimated_reduction_pct:.0f}%")
print()

print(f"Optimization Candidates: {len(candidates)}")
for i, c in enumerate(candidates[:5], 1):
    print(f"  {i}. {c.optimization_type.value}")
    print(f"     Target: {c.target}")
    print(f"     Expected Traffic Reduction: {c.expected_traffic_reduction_pct:.0f}%")
    print(f"     Expected Speedup: {c.expected_speedup:.2f}x")
    print(f"     Priority: {c.priority:.1f}")
print()


# ============================================================
# STEP 3: ADAPTIVE OPTIMIZER
# ============================================================
print("=" * 70)
print("STEP 3: ADAPTIVE OPTIMIZATION + FEEDBACK LOOP")
print("=" * 70)
print()

from memopt.profiler import AdaptiveOptimizer

optimizer = AdaptiveOptimizer()

if candidates:
    print("Applying optimizations with test-measure-commit loop...")
    print()

    session = optimizer.optimize(
        model=model,
        candidates=candidates[:5],  # Top 5 candidates
        input_fn=lambda: torch.randn(batch_size, seq_len, 512).to(device),
        num_warmup=3,
        num_measure=5,
        min_improvement=0.005  # 0.5% minimum improvement
    )

    print(f"Committed Optimizations: {session.committed_count}")
    print(f"Rolled Back: {session.rollback_count}")
    print(f"Total Speedup: {session.total_speedup:.3f}x")
    print(f"Total Traffic Reduction: {session.total_traffic_reduction_pct:.1f}%")
    print()

    print("Optimization Results:")
    for i, r in enumerate(session.results, 1):
        status_icon = "✓" if r.status.value == "committed" else "✗"
        print(f"  {i}. [{status_icon}] {r.candidate.optimization_type.value}")
        print(f"     Status: {r.status.value}")
        if r.status.value == "committed":
            print(f"     Actual Speedup: {r.actual_speedup:.3f}x")
            print(f"     Baseline: {r.baseline_time_ms:.2f}ms → Optimized: {r.optimized_time_ms:.2f}ms")
        elif r.failure_reason:
            print(f"     Reason: {r.failure_reason[:60]}...")
        print(f"     Semantics Verified: {r.semantics_verified}")
else:
    print("No optimization candidates found.")

print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print()

print("Pipeline Results:")
print(f"  Step 1 - Profiling: {len(snapshot.kernel_analyses)} regions profiled")
print(f"  Step 2 - Attribution: {len(candidates)} optimizations identified")
if candidates:
    print(f"  Step 3 - Optimization: {session.total_speedup:.2f}x speedup achieved")

print()
print("What We Can Claim:")
print(f"  ✓ Profiled {len(snapshot.kernel_analyses)} forward passes")
print(f"  ✓ Tracked {snapshot.total_dram_bytes / 1024**2:.1f} MB DRAM traffic")
print(f"  ✓ Identified {len(candidates)} optimization opportunities")
if candidates and session.committed_count > 0:
    print(f"  ✓ Applied {session.committed_count} optimizations with verified semantics")
    print(f"  ✓ Achieved {session.total_speedup:.2f}x measurable speedup")

print()
print("=" * 70)
print("INTEGRATION TEST COMPLETE")
print("=" * 70)
