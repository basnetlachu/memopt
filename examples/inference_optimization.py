#!/usr/bin/env python3
"""
Inference Optimization Example

Demonstrates memory access coalescing for LLM inference.
Measures real GPU bandwidth reduction during text generation.

Usage:
    python examples/inference_optimization.py

Requirements:
    - PyTorch with CUDA
    - transformers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memopt.optimization import MemoryCoalescer, CoalescingConfig
from memopt.measurement import BandwidthTracker


def run_baseline(model, tokenizer, prompts: list, max_tokens: int = 50):
    """Run inference without optimization."""
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        results.append(outputs)
    return results


def run_optimized(model, tokenizer, coalescer, prompts: list, max_tokens: int = 50):
    """Run inference with memory coalescing enabled."""
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        results.append(outputs)
    return results


def main():
    print("=" * 70)
    print("INFERENCE OPTIMIZATION DEMO")
    print("=" * 70)
    print()

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device == 'cpu':
        print("⚠️  Warning: Running on CPU. For real bandwidth measurement, use GPU.")
        print()

    # Load model
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    print(f"Model loaded: GPT-2 ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    print()

    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Machine learning algorithms can",
        "Deep neural networks are",
        "Natural language processing enables",
        "Computer vision technology allows",
    ] * 4  # 20 prompts total

    max_tokens = 50

    print(f"Workload: {len(prompts)} prompts × {max_tokens} max tokens")
    print()

    # Initialize tracking
    tracker = BandwidthTracker()

    # Warm up
    print("Warming up...")
    _ = run_baseline(model, tokenizer, prompts[:2], max_tokens=10)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print()

    # Baseline measurement
    print("[1/2] Measuring BASELINE (no optimization)...")
    with tracker.measure("baseline"):
        baseline_results = run_baseline(model, tokenizer, prompts, max_tokens)

    baseline = tracker.get_measurement("baseline")
    print(f"      Peak memory: {baseline.peak_memory_gb:.3f} GB")
    print(f"      Duration: {baseline.duration_ms:.1f} ms")
    print()

    # Initialize coalescer
    config = CoalescingConfig(
        cache_size_mb=256,
        optimize_attention=True,
        optimize_mlp=True,
        enable_profiling=True
    )
    coalescer = MemoryCoalescer(model, mode='inference', config=config)

    # Optimized measurement
    print("[2/2] Measuring OPTIMIZED (with memory coalescing)...")
    coalescer.enable()

    with tracker.measure("optimized"):
        optimized_results = run_optimized(
            model, tokenizer, coalescer, prompts, max_tokens
        )

    optimized = tracker.get_measurement("optimized")
    print(f"      Peak memory: {optimized.peak_memory_gb:.3f} GB")
    print(f"      Duration: {optimized.duration_ms:.1f} ms")
    print()

    # Get coalescing stats
    coalescer_stats = coalescer.get_stats()
    coalescer.disable()

    # Generate comparison report
    report = tracker.compare("baseline", "optimized")

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("Memory Measurements:")
    print(f"  Baseline peak:   {report.baseline.peak_memory_gb:.3f} GB")
    print(f"  Optimized peak:  {report.optimized.peak_memory_gb:.3f} GB")
    print(f"  Reduction:       {report.memory_reduction_pct:.1f}%")
    print()

    print("Timing:")
    print(f"  Baseline:   {report.baseline.duration_ms:.1f} ms")
    print(f"  Optimized:  {report.optimized.duration_ms:.1f} ms")
    print(f"  Speedup:    {report.speedup_ratio:.2f}x")
    print()

    print("Coalescing Statistics:")
    print(f"  Layers optimized:  {coalescer_stats.layers_optimized}")
    print(f"  Total accesses:    {coalescer_stats.total_accesses:,}")
    print(f"  Cache hits:        {coalescer_stats.cache_hits:,}")
    print(f"  Cache misses:      {coalescer_stats.cache_misses:,}")
    print(f"  Hit rate:          {coalescer_stats.hit_rate:.1f}%")
    print(f"  Bandwidth reduction: {coalescer_stats.bandwidth_reduction:.1f}%")
    print()

    # Verify correctness
    print("Correctness Check:")
    correct = all(
        torch.equal(b, o)
        for b, o in zip(baseline_results, optimized_results)
    )
    if correct:
        print("  ✅ Outputs match (optimization is lossless)")
    else:
        print("  ❌ Outputs differ (check implementation)")
    print()

    print("=" * 70)

    # Summary
    if report.memory_reduction_pct >= 15:
        print(f"✅ SUCCESS: {report.memory_reduction_pct:.1f}% memory reduction")
        print("   Ready for production use")
    elif report.memory_reduction_pct >= 5:
        print(f"⚠️  PARTIAL: {report.memory_reduction_pct:.1f}% memory reduction")
        print("   Some improvement, may need tuning")
    else:
        print(f"❌ MINIMAL: {report.memory_reduction_pct:.1f}% memory reduction")
        print("   Workload may not benefit from coalescing")

    print("=" * 70)

    return report


if __name__ == '__main__':
    main()
