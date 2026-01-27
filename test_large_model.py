#!/usr/bin/env python3
"""
Large Model Integration Test - Full 3-Step Pipeline

Tests with a large transformer model (GPT-2 style, 350M+ params)
to validate all optimization steps under real workload.
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# Results accumulator
results = {
    "test_date": datetime.now().isoformat(),
    "environment": {},
    "model_info": {},
    "step1_profiling": {},
    "step2_attribution": {},
    "step3_optimization": {},
    "summary": {}
}

print("=" * 70)
print("LARGE MODEL INTEGRATION TEST")
print("=" * 70)
print()

# Environment info
print("ENVIRONMENT")
print("-" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

results["environment"]["pytorch_version"] = torch.__version__
results["environment"]["cuda_available"] = torch.cuda.is_available()

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_mem_gb:.1f} GB")
    results["environment"]["gpu_name"] = gpu_name
    results["environment"]["gpu_memory_gb"] = round(gpu_mem_gb, 1)
print()


# Build a GPT-2 style transformer (350M+ params)
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT2StyleModel(nn.Module):
    """GPT-2 Medium style model (~350M params)"""
    def __init__(self, vocab_size=50257, dim=1024, depth=24, heads=16, max_seq=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)

        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.head(x)
        return logits


# Create model
print("MODEL")
print("-" * 70)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPT-2 Medium config: dim=1024, depth=24, heads=16
model = GPT2StyleModel(dim=1024, depth=24, heads=16).to(device)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model: GPT-2 Medium Style")
print(f"Parameters: {param_count / 1e6:.1f}M ({param_count:,})")
print(f"Device: {device}")

results["model_info"]["name"] = "GPT-2 Medium Style"
results["model_info"]["parameters"] = param_count
results["model_info"]["parameters_millions"] = round(param_count / 1e6, 1)
results["model_info"]["config"] = {"dim": 1024, "depth": 24, "heads": 16}
print()

# Sample input (batch=4, seq_len=512)
batch_size = 4
seq_len = 512
sample_input = torch.randint(0, 50257, (batch_size, seq_len)).to(device)

print(f"Input: batch_size={batch_size}, seq_len={seq_len}")
print()


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

# Profile 10 forward passes
num_profile_iters = 10
for i in range(num_profile_iters):
    with profiler.profile_region(f"forward_{i}"):
        with torch.no_grad():
            output = model(sample_input)
        if device == 'cuda':
            torch.cuda.synchronize()

profiler.stop()
snapshot = profiler.snapshot()

print(f"Profiled Iterations: {num_profile_iters}")
print(f"Total GPU Time: {snapshot.total_gpu_time_ms:.2f} ms")
print(f"Avg Time per Forward: {snapshot.total_gpu_time_ms / num_profile_iters:.2f} ms")
print(f"Total DRAM Traffic: {snapshot.total_dram_bytes / 1024**2:.2f} MB")
print(f"Memory-Bound Time: {snapshot.memory_bound_pct:.1f}%")
print()

results["step1_profiling"]["profiled_iterations"] = num_profile_iters
results["step1_profiling"]["total_gpu_time_ms"] = round(snapshot.total_gpu_time_ms, 2)
results["step1_profiling"]["avg_time_per_forward_ms"] = round(snapshot.total_gpu_time_ms / num_profile_iters, 2)
results["step1_profiling"]["total_dram_traffic_mb"] = round(snapshot.total_dram_bytes / 1024**2, 2)
results["step1_profiling"]["memory_bound_pct"] = round(snapshot.memory_bound_pct, 1)

print("Top Bottlenecks:")
bottlenecks_info = []
for i, b in enumerate(snapshot.top_bottlenecks[:5], 1):
    print(f"  {i}. {b.kernel_name}")
    print(f"     Type: {b.bottleneck_type.value}")
    print(f"     Impact Score: {b.impact_score:.4f}")
    print(f"     Memory Stall: {b.memory_stall_pct:.1f}%")
    bottlenecks_info.append({
        "kernel": b.kernel_name,
        "type": b.bottleneck_type.value,
        "impact_score": round(b.impact_score, 4),
        "memory_stall_pct": round(b.memory_stall_pct, 1)
    })
results["step1_profiling"]["top_bottlenecks"] = bottlenecks_info
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

# Need tensor input for attribution
sample_tensor_input = torch.randn(batch_size, seq_len, 1024).to(device)

# For attribution, we'll create a simpler wrapper
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed = model.tok_emb
        self.blocks = model.blocks
        self.norm = model.norm

    def forward(self, x):
        # x is already embedded tensor
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

wrapper = ModelWrapper(model)
wrapper.eval()

attributor.analyze_model(wrapper, sample_tensor_input)

attributions = attributor.get_attributions()
candidates = attributor.get_optimization_candidates()

print(f"Attributions Found: {len(attributions)}")
attributions_info = []
for i, attr in enumerate(attributions[:5], 1):
    print(f"  {i}. {attr.attribution_type.value}")
    print(f"     Confidence: {attr.confidence:.0%}")
    print(f"     Affected Tensors: {len(attr.affected_tensors)}")
    print(f"     Estimated Reduction: {attr.estimated_reduction_pct:.0f}%")
    print(f"     Traffic Impact: {attr.estimated_traffic_bytes / 1024**2:.1f} MB")
    attributions_info.append({
        "type": attr.attribution_type.value,
        "confidence": round(attr.confidence, 2),
        "affected_tensors": len(attr.affected_tensors),
        "estimated_reduction_pct": round(attr.estimated_reduction_pct, 0),
        "traffic_impact_mb": round(attr.estimated_traffic_bytes / 1024**2, 1)
    })
results["step2_attribution"]["total_attributions"] = len(attributions)
results["step2_attribution"]["attributions"] = attributions_info
print()

print(f"Optimization Candidates: {len(candidates)}")
candidates_info = []
for i, c in enumerate(candidates[:10], 1):
    print(f"  {i}. {c.optimization_type.value}")
    print(f"     Target: {c.target[:50]}...")
    print(f"     Expected Traffic Reduction: {c.expected_traffic_reduction_pct:.0f}%")
    print(f"     Expected Speedup: {c.expected_speedup:.2f}x")
    print(f"     Priority: {c.priority:.1f}")
    candidates_info.append({
        "type": c.optimization_type.value,
        "target": c.target[:50],
        "expected_reduction_pct": round(c.expected_traffic_reduction_pct, 0),
        "expected_speedup": round(c.expected_speedup, 2),
        "priority": round(c.priority, 1)
    })
results["step2_attribution"]["total_candidates"] = len(candidates)
results["step2_attribution"]["candidates"] = candidates_info
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
    print(f"Using {optimizer.measurer.num_measure} measurement iterations")
    print()

    session = optimizer.optimize(
        model=wrapper,
        candidates=candidates[:5],  # Top 5 candidates
        input_fn=lambda: torch.randn(batch_size, seq_len, 1024).to(device),
        num_warmup=5,
        num_measure=20,
        min_improvement=0.005  # 0.5% minimum
    )

    print(f"Committed Optimizations: {session.committed_count}")
    print(f"Rolled Back: {session.rollback_count}")
    print(f"Total Speedup: {session.total_speedup:.3f}x")
    print(f"Total Traffic Reduction: {session.total_traffic_reduction_pct:.1f}%")
    print()

    results["step3_optimization"]["committed"] = session.committed_count
    results["step3_optimization"]["rolled_back"] = session.rollback_count
    results["step3_optimization"]["total_speedup"] = round(session.total_speedup, 3)
    results["step3_optimization"]["total_traffic_reduction_pct"] = round(session.total_traffic_reduction_pct, 1)

    print("Optimization Results:")
    opt_results = []
    for i, r in enumerate(session.results, 1):
        status_icon = "✓" if r.status.value == "committed" else "✗"
        print(f"  {i}. [{status_icon}] {r.candidate.optimization_type.value}")
        print(f"     Status: {r.status.value}")

        result_info = {
            "type": r.candidate.optimization_type.value,
            "status": r.status.value,
            "semantics_verified": r.semantics_verified
        }

        if r.status.value == "committed":
            print(f"     Actual Speedup: {r.actual_speedup:.3f}x")
            print(f"     Baseline: {r.baseline_time_ms:.2f}ms → Optimized: {r.optimized_time_ms:.2f}ms")
            if r.baseline_metrics and r.optimized_metrics:
                print(f"     Measurement CV: baseline={r.baseline_metrics.coefficient_of_variation:.1f}%, opt={r.optimized_metrics.coefficient_of_variation:.1f}%")
            result_info["actual_speedup"] = round(r.actual_speedup, 3)
            result_info["baseline_ms"] = round(r.baseline_time_ms, 2)
            result_info["optimized_ms"] = round(r.optimized_time_ms, 2)
        elif r.failure_reason:
            print(f"     Reason: {r.failure_reason[:80]}")
            result_info["failure_reason"] = r.failure_reason[:100]

        print(f"     Semantics Verified: {r.semantics_verified}")
        opt_results.append(result_info)

    results["step3_optimization"]["results"] = opt_results
else:
    print("No optimization candidates found.")
    results["step3_optimization"]["committed"] = 0
    results["step3_optimization"]["rolled_back"] = 0
    results["step3_optimization"]["total_speedup"] = 1.0

print()


# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print()

# Calculate overall results
committed = results["step3_optimization"].get("committed", 0)
speedup = results["step3_optimization"].get("total_speedup", 1.0)

print("Pipeline Results:")
print(f"  Step 1 - Profiling: {results['step1_profiling']['profiled_iterations']} iterations profiled")
print(f"  Step 1 - Avg forward time: {results['step1_profiling']['avg_time_per_forward_ms']:.2f} ms")
print(f"  Step 2 - Attribution: {results['step2_attribution']['total_candidates']} optimizations identified")
print(f"  Step 3 - Optimization: {speedup:.3f}x speedup achieved ({committed} committed)")
print()

results["summary"]["step1_worked"] = results["step1_profiling"]["profiled_iterations"] > 0
results["summary"]["step2_worked"] = results["step2_attribution"]["total_candidates"] > 0
results["summary"]["step3_worked"] = committed > 0 or results["step3_optimization"].get("rolled_back", 0) > 0
results["summary"]["final_speedup"] = speedup

print("Honest Assessment:")
if speedup > 1.1:
    print(f"  ✓ Achieved meaningful speedup: {speedup:.3f}x")
elif speedup > 1.0:
    print(f"  ~ Achieved minor speedup: {speedup:.3f}x")
else:
    print(f"  ✗ No speedup achieved: {speedup:.3f}x")

if committed > 0:
    print(f"  ✓ {committed} optimizations passed semantic verification and improved performance")
else:
    print(f"  ✗ No optimizations met the improvement threshold")

rolled_back = results["step3_optimization"].get("rolled_back", 0)
if rolled_back > 0:
    print(f"  ⚠ {rolled_back} optimizations rolled back (didn't improve or changed semantics)")

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)

# Save results to JSON
results_file = "validation/large_model_results.json"
os.makedirs("validation", exist_ok=True)
with open(results_file, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {results_file}")
